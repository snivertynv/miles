"""In-memory span export for the observer tests; nemo-lens is imported only by the tests that need it."""

import functools
import os

import pytest


def _reset_otel():
    import nemo.lens.handle
    import opentelemetry.metrics._internal as metrics_internal
    import opentelemetry.trace as trace_module
    from opentelemetry.util._once import Once

    trace_module._TRACER_PROVIDER = None
    trace_module._TRACER_PROVIDER_SET_ONCE = Once()
    metrics_internal._METER_PROVIDER = None
    metrics_internal._METER_PROVIDER_SET_ONCE = Once()
    nemo.lens.handle._INITIALIZED = False


@pytest.fixture
def exporter(monkeypatch):
    """Every span the observer exports in this test, as an in-memory list."""
    import miles_lens.observer as observer_module
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    class MemoryExporter(SpanExporter):
        def __init__(self):
            self.spans = []

        def export(self, spans):
            self.spans.extend(spans)
            return SpanExportResult.SUCCESS

        def shutdown(self): ...

        def force_flush(self, timeout_millis=0):
            return True

    for key in [k for k in os.environ if k.startswith(("MILES_LENS_", "NEMO_LENS_", "OTEL_"))]:
        monkeypatch.delenv(key)
    monkeypatch.delenv("ENABLE_SINGLE_TRACEID", raising=False)
    monkeypatch.setenv("MILES_LENS_ENABLED", "1")
    monkeypatch.setenv("MILES_LENS_METRICS_ENABLED", "0")
    memory = MemoryExporter()
    monkeypatch.setattr(
        observer_module, "setup_telemetry", functools.partial(observer_module.setup_telemetry, span_exporter=memory)
    )
    _reset_otel()
    yield memory
    _reset_otel()
