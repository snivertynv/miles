"""Spans reach observers as plain data, cost nothing without a listener, and never change what they wrap."""

import asyncio
import inspect
from types import SimpleNamespace

import pytest

from miles.utils import observers as observers_module
from miles.utils.misc import SingletonMeta
from miles.utils.observers import Observer, Observers, SpanEnd


class _SpanRecorder(Observer):
    def __init__(self, device: bool = False):
        self.device = device
        self.starts: list = []
        self.ends: list = []

    def span_start(self, span):
        self.starts.append(span)

    def span_end(self, span):
        self.ends.append(span)

    def wants_device_spans(self):
        return self.device


class _PhasesOnly(Observer):
    def phase_end(self, name, t0, t1): ...


@pytest.fixture
def install(monkeypatch):
    def _install(*observers: Observer) -> Observers:
        SingletonMeta._instances.pop(Observers, None)
        installed = [SimpleNamespace(name=type(o).__name__, load=lambda o=o: lambda: o) for o in observers]
        monkeypatch.setattr(observers_module, "entry_points", lambda group: installed)
        return Observers()

    yield _install
    SingletonMeta._instances.pop(Observers, None)


def test_span_reports_start_and_end_with_plain_attributes(install):
    recorder = _SpanRecorder()
    install(recorder)

    with observers_module.span("rollout.dispatch", category="dispatch", rollout_id=3, path=None, shape=(2, 4)):
        pass

    [start], [end] = recorder.starts, recorder.ends
    assert start["name"] == end["name"] == "rollout.dispatch"
    assert start["span_id"] == end["span_id"]
    assert end["category"] == "dispatch"
    assert end["attributes"] == {"rollout_id": 3, "shape": "(2, 4)"}
    assert start["t0"] <= end["t1"]
    assert end["error"] == ""


def test_span_records_the_error_and_reraises(install):
    recorder = _SpanRecorder()
    install(recorder)

    with pytest.raises(ValueError), observers_module.span("engine.offload", category="offload"):
        raise ValueError("boom")

    assert recorder.ends[0]["error"] == "ValueError"


def test_observed_takes_rollout_id_from_positional_or_keyword_arguments(install):
    recorder = _SpanRecorder()
    install(recorder)

    class Actor:
        @observers_module.observed(category="train")
        def train(self, rollout_id, data):
            return data

        @observers_module.observed("engine.prepare", category="engine")
        async def prepare(self, rollout_id=None):
            await asyncio.sleep(0)
            return "ready"

    assert Actor().train(5, data="d") == "d"
    assert asyncio.run(Actor().prepare(rollout_id=7)) == "ready"
    assert asyncio.run(Actor().prepare()) == "ready"

    assert [(e["name"], e["attributes"]) for e in recorder.ends] == [
        ("train", {"rollout_id": 5}),
        ("engine.prepare", {"rollout_id": 7}),
        ("engine.prepare", {}),
    ]


def test_observed_keeps_the_wrapped_signature(install):
    install()

    @observers_module.observed(category="train")
    def forward_step(data_iterator, model, return_schedule_plan: bool = False): ...

    assert "return_schedule_plan" in inspect.signature(forward_step).parameters
    assert inspect.iscoroutinefunction(observers_module.observed(category="engine")(_async_noop))


async def _async_noop(): ...


def test_nothing_is_reported_without_a_span_listener(install):
    observers = install(_PhasesOnly())
    assert not observers.hears_spans

    with observers_module.span("driver.step", category="step", rollout_id=1):
        pass
    assert observers_module.observed(category="train")(lambda rollout_id: rollout_id + 1)(1) == 2
    assert not observers.wants_device_spans()


def test_report_spans_delivers_ends_without_starts(install):
    recorder = _SpanRecorder(device=True)
    observers = install(recorder)
    span = SpanEnd(
        span_id=observers_module.new_span_id(),
        name="forward",
        category="gpu",
        t0=1.0,
        t1=2.0,
        attributes={"microbatch": 0},
        error="",
    )

    observers_module.report_spans([span])

    assert recorder.starts == [] and recorder.ends == [span]
    assert observers.wants_device_spans()
