"""``LensObserver``: the stages of a miles run as nemo-lens spans on one timeline.

Configured like any nemo-lens consumer, from ``MILES_LENS_*`` variables that fall
back to ``NEMO_LENS_*`` (``ENABLED``, ``SPAN_GROUPS``, ``EXPORTER``, ...) and the
standard ``OTEL_EXPORTER_OTLP_*`` ones. The driver forwards them to every worker
it launches, together with a run id and ``ENABLE_SINGLE_TRACEID``, so every
process exports into one trace: one timeline per job, one lane per process.

``MILES_LENS_TRAIN_RANKS`` limits which training ranks export (``all``, the
default, or a comma-separated list), since their spans multiply with world size.
"""

import logging
import os
from typing import Any

from miles_lens.groups import GPU_GROUP
from miles_lens.samples import SampleSpans
from miles_lens.timers import timer_group
from nemo.lens import NemoLensConfig, TelemetryHandle, is_span_group_enabled, setup_telemetry
from nemo.lens.semconv import NV_DL_RANK, NV_DL_ROLE
from nemo.lens.span_utilities import emit_span

from miles.utils.observers import Observer, ProcessStart, RunStart, SpanEnd, SpanStart, TrajectoryEvent

logger = logging.getLogger(__name__)

ENV_PREFIX = "MILES_LENS"
_FALLBACK_PREFIX = "NEMO_LENS"
TRAIN_RANKS_ENV = f"{ENV_PREFIX}_TRAIN_RANKS"
_FORWARDED_PREFIXES = ("MILES_LENS_", "NEMO_LENS_", "OTEL_")
# workers' environment is logged and kept in Ray job metadata; OTLP headers carry credentials
_SECRET_MARKERS = ("HEADERS", "TOKEN", "KEY", "PASSWORD", "SECRET")


class LensObserver(Observer):
    def __init__(self):
        self._config = _config_from_env()
        self._handle: TelemetryHandle | None = None
        self._rollout_id: int | None = None
        self._samples = SampleSpans(self._emit)

    def enabled(self) -> bool:
        return self._config.enabled and self._config.traces_enabled

    def start_run(self, run: RunStart) -> dict[str, str]:
        env = {
            key: value
            for key, value in os.environ.items()
            if key.startswith(_FORWARDED_PREFIXES) and not any(marker in key for marker in _SECRET_MARKERS)
        }
        if not (env.get(f"{ENV_PREFIX}_RUN_ID") or env.get(f"{_FALLBACK_PREFIX}_RUN_ID")):
            env[f"{ENV_PREFIX}_RUN_ID"] = run["run_uuid"]
        env.setdefault("ENABLE_SINGLE_TRACEID", os.environ.get("ENABLE_SINGLE_TRACEID", "1"))
        # the driver's own start_process comes next and reads the same settings
        os.environ.update(env)
        self._config = _config_from_env()
        return env

    def start_process(self, process: ProcessStart) -> None:
        role, rank = process["role"], process["rank"]
        if role == "train" and not _train_rank_selected(rank):
            return
        self._config.service_name = f"{os.environ.get('OTEL_SERVICE_NAME') or 'miles'}.{role}"
        self._handle = setup_telemetry(
            self._config,
            resource_attributes={
                NV_DL_ROLE: role,
                NV_DL_RANK: rank,
                "miles.node": process["node"],
                "miles.gpus": ",".join(str(gpu) for gpu in process["gpus"]),
            },
        )

    def wants_device_spans(self) -> bool:
        return self._handle is not None and is_span_group_enabled(GPU_GROUP)

    def phase_end(self, name: str, t0: float, t1: float) -> None:
        self._emit(name, t0, t1, group=timer_group(name), attributes={"miles.source": "timer"})

    def span_start(self, span: SpanStart) -> None:
        # timer phases and samples carry no step; they take the one of the work last begun here
        if (rollout_id := span["attributes"].get("rollout_id")) is not None:
            self._rollout_id = rollout_id

    def span_end(self, span: SpanEnd) -> None:
        attributes = {f"miles.{key}": value for key, value in span["attributes"].items()}
        if span["error"]:
            attributes["error.type"] = span["error"]
        self._emit(span["name"], span["t0"], span["t1"], group=span["category"], attributes=attributes)

    def trajectory_event(self, event: TrajectoryEvent) -> None:
        self._samples.on_event(event)

    def end_process(self) -> None:
        if self._handle is not None:
            self._handle.shutdown()

    def end_run(self, *, succeeded: bool) -> None:
        self.end_process()

    def _emit(self, name: str, t0: float, t1: float, *, group: str, attributes: dict[str, Any]) -> None:
        if self._handle is None:
            return
        attributes = {**attributes, "miles.group": group}
        if "miles.rollout_id" not in attributes and self._rollout_id is not None:
            attributes["miles.rollout_id"] = self._rollout_id
        emit_span(self._handle.tracer, name, t0, t1, group=group, attributes=attributes)


def _config_from_env() -> NemoLensConfig:
    return NemoLensConfig.from_env(prefix=ENV_PREFIX, fallback_prefix=_FALLBACK_PREFIX)


def _train_rank_selected(rank: int) -> bool:
    spec = os.environ.get(TRAIN_RANKS_ENV, "all").strip()
    if spec in ("", "all"):
        return True
    return rank in {int(part) for part in spec.split(",") if part.strip()}
