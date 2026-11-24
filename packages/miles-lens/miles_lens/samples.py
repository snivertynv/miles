"""Per-sample spans from the rollout manager's trajectory events."""

from collections.abc import Callable
from typing import Any

from miles_lens.groups import SAMPLE_GROUP

from miles.utils.observers import TrajectoryEvent

Emit = Callable[..., None]


class SampleSpans:
    """Pairs each start event with its end and emits the span between them.

    An attempt yields ``sample.queue`` (waiting for a generation slot), one
    ``sample.generate`` per generation call, one ``sample.tool`` per tool call or
    agent gap, and ``sample.attempt`` around all of them.
    """

    # starts whose end never came (aborted samples) are dropped oldest first past this
    MAX_OPEN = 100_000

    def __init__(self, emit: Emit):
        self._emit = emit
        self._open: dict[tuple, float] = {}

    def on_event(self, event: TrajectoryEvent) -> None:
        sample = (event["group_index"], event["sample_index"])
        kind, ts, turn = event["kind"], event["ts"], event["turn"]
        attributes = _attributes_of(event)
        if kind == "attempt_start":
            self._put(("attempt", sample), ts)
            self._put(("queue", sample), ts)
        elif kind == "gen_start":
            self._end("sample.queue", ("queue", sample), ts, attributes)
            self._put(("gen", sample, turn), ts)
        elif kind == "gen_end":
            self._end(
                "sample.generate",
                ("gen", sample, turn),
                ts,
                {**attributes, "miles.completion_tokens": event["detail"]},
            )
        elif kind == "tool_start":
            self._put(("tool", sample, turn), ts)
        elif kind == "tool_end":
            self._end("sample.tool", ("tool", sample, turn), ts, {**attributes, "miles.tool": event["detail"]})
        elif kind == "attempt_end":
            self._open.pop(("queue", sample), None)
            self._end("sample.attempt", ("attempt", sample), ts, {**attributes, "miles.status": event["detail"]})

    def _put(self, key: tuple, ts: float) -> None:
        self._open[key] = ts
        if len(self._open) > self.MAX_OPEN:
            del self._open[next(iter(self._open))]

    def _end(self, name: str, key: tuple, ts: float, attributes: dict[str, Any]) -> None:
        start = self._open.pop(key, None)
        if start is not None:
            self._emit(name, start, ts, group=SAMPLE_GROUP, attributes=attributes)


def _attributes_of(event: TrajectoryEvent) -> dict[str, Any]:
    attributes = {"miles.sample_index": event["sample_index"], "miles.group_index": event["group_index"]}
    if event["turn"] >= 0:
        attributes["miles.turn"] = event["turn"]
    if event["weight_version"]:
        attributes["miles.weight_version"] = event["weight_version"]
    return attributes
