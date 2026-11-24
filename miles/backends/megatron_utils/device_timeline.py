"""Device-time spans of Megatron training, for the run observers.

Regions are bracketed by timing events on the current stream, so a span is when
the device ran the region's work, not when the host queued it. Events are read
back once per training call: ``collect`` synchronizes at its start to anchor the
device clock to wall time, and synchronizes again at its end to convert and
report every region. Only the Megatron backend uses this, so it runs on CUDA.
"""

import functools
import itertools
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch

from miles.platforms import current_platform
from miles.utils import observers
from miles.utils.misc import SingletonMeta


@dataclass
class _Region:
    name: str
    start: torch.Event
    end: torch.Event
    attributes: dict[str, Any]


@dataclass
class _Collection:
    anchor: torch.Event
    anchor_wall: float
    attributes: dict[str, Any]
    regions: list[_Region] = field(default_factory=list)
    microbatches: Iterator[int] = field(default_factory=itertools.count)


class DeviceTimeline(metaclass=SingletonMeta):
    def __init__(self):
        self.enabled = observers.Observers().wants_device_spans()
        self._collection: _Collection | None = None

    @contextmanager
    def collect(self, **attributes: Any) -> Iterator[None]:
        """Time the regions of one training call; nested calls join the outer one."""
        if not self.enabled or self._collection is not None:
            yield
            return
        torch.accelerator.synchronize()
        anchor = _new_event()
        anchor.record()
        self._collection = _Collection(anchor=anchor, anchor_wall=time.time(), attributes=attributes)
        try:
            yield
        finally:
            collection, self._collection = self._collection, None
            torch.accelerator.synchronize()
            observers.report_spans(_spans_of(collection))

    def region(self, name: str, **attributes: Any):
        if self._collection is None or torch.cuda.is_current_stream_capturing():
            return nullcontext()
        return self._region(name, attributes)

    def microbatch(self, name: str):
        """A region numbered by its order within the collection."""
        if self._collection is None:
            return nullcontext()
        return self.region(name, microbatch=next(self._collection.microbatches))

    def wrap_forward_step(self, forward_step):
        """Time each call of a Megatron forward step as one microbatch's forward."""
        if not self.enabled:
            return forward_step

        @functools.wraps(forward_step)
        def wrapper(*args, **kwargs):
            with self.microbatch("forward"):
                return forward_step(*args, **kwargs)

        return wrapper

    @contextmanager
    def _region(self, name: str, attributes: dict[str, Any]) -> Iterator[None]:
        start, end = _new_event(), _new_event()
        start.record()
        try:
            yield
        finally:
            end.record()
            self._collection.regions.append(_Region(name=name, start=start, end=end, attributes=attributes))


def _new_event() -> torch.Event:
    return torch.Event(device=current_platform.device_type, enable_timing=True)


def _spans_of(collection: _Collection) -> list[observers.SpanEnd]:
    def wall(event: torch.Event) -> float:
        return collection.anchor_wall + collection.anchor.elapsed_time(event) / 1000.0

    return [
        observers.SpanEnd(
            span_id=observers.new_span_id(),
            name=region.name,
            category="gpu",
            t0=wall(region.start),
            t1=wall(region.end),
            attributes={**collection.attributes, **region.attributes},
            error="",
        )
        for region in collection.regions
    ]
