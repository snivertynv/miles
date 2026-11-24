"""miles reports become nemo-lens spans: named, timed and tagged as miles reported them."""

import os

import pytest

# CI does not install nemo-lens; these skip there, the converter tests still run
pytest.importorskip("nemo.lens")
pytest.importorskip("opentelemetry.sdk")

from miles_lens.observer import LensObserver  # noqa: E402


def _process(role="train", rank=0):
    return {"role": role, "node": "10.0.0.1", "rank": rank, "gpus": [0, 1]}


def _span(span_id, name, category, t0, t1, error="", **attributes):
    return {
        "span_id": span_id,
        "name": name,
        "category": category,
        "t0": t0,
        "t1": t1,
        "attributes": attributes,
        "error": error,
    }


def _finished(observer, exporter):
    observer.end_process()
    return {span.name: span for span in exporter.spans}


def test_disabled_without_the_switch(monkeypatch, exporter):
    monkeypatch.setenv("MILES_LENS_ENABLED", "0")
    assert not LensObserver().enabled()


def test_start_run_forwards_settings_and_names_the_run(monkeypatch, exporter):
    monkeypatch.setenv("NEMO_LENS_SPAN_GROUPS", "per_sample")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "authorization=Bearer secret")

    env = LensObserver().start_run({"run_uuid": "run-1", "start_ts": 0.0, "args": {}})

    assert env["NEMO_LENS_SPAN_GROUPS"] == "per_sample"
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://collector:4317"
    assert env["MILES_LENS_RUN_ID"] == "run-1"
    assert env["ENABLE_SINGLE_TRACEID"] == "1"
    assert "OTEL_EXPORTER_OTLP_HEADERS" not in env
    assert os.environ["MILES_LENS_RUN_ID"] == "run-1"


def test_a_run_id_the_user_set_is_kept(monkeypatch, exporter):
    monkeypatch.setenv("NEMO_LENS_RUN_ID", "mine")
    env = LensObserver().start_run({"run_uuid": "run-1", "start_ts": 0.0, "args": {}})
    assert "MILES_LENS_RUN_ID" not in env and env["NEMO_LENS_RUN_ID"] == "mine"


def test_spans_and_timer_phases_share_the_step_and_the_process_identity(exporter):
    observer = LensObserver()
    observer.start_process(_process(rank=3))
    observer.span_start(_span(1, "rank.train_step", "step", 10.0, 10.0, rollout_id=4))
    observer.phase_end("actor_train", 11.0, 15.5)
    observer.phase_end("actor/rollout", 9.0, 10.0)
    observer.phase_end("some_new_timer", 15.0, 16.0)
    observer.span_end(_span(1, "rank.train_step", "step", 10.0, 16.0, rollout_id=4))

    spans = _finished(observer, exporter)

    train = spans["actor_train"]
    assert (train.start_time, train.end_time) == (11_000_000_000, 15_500_000_000)
    assert train.attributes["miles.rollout_id"] == 4
    assert train.attributes["miles.source"] == "timer"
    assert spans["rank.train_step"].attributes["miles.rollout_id"] == 4
    assert "actor/rollout" in spans and "some_new_timer" in spans
    resource = train.resource.attributes
    assert resource["nv.dl.role"] == "train" and resource["nv.dl.rank"] == 3
    assert resource["service.name"] == "miles.train"
    assert len({span.context.trace_id for span in exporter.spans}) == len(exporter.spans)  # no single trace asked


def test_failed_work_is_marked(exporter):
    observer = LensObserver()
    observer.start_process(_process(role="inference_controller", rank=-1))
    observer.span_end(_span(7, "engine.offload", "offload", 1.0, 2.0, error="TimeoutError"))
    assert _finished(observer, exporter)["engine.offload"].attributes["error.type"] == "TimeoutError"


def test_single_trace_id_puts_every_span_in_one_trace(monkeypatch, exporter):
    monkeypatch.setenv("ENABLE_SINGLE_TRACEID", "1")
    monkeypatch.setenv("MILES_LENS_RUN_ID", "run-1")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    observer = LensObserver()
    observer.start_process(_process())
    observer.phase_end("actor_train", 1.0, 2.0)
    observer.span_end(_span(1, "rank.init", "init", 0.0, 1.0))

    _finished(observer, exporter)
    assert len({span.context.trace_id for span in exporter.spans}) == 1


@pytest.mark.parametrize(("spec", "exports"), [("default", False), ("per_sample", True)])
def test_sample_spans_are_opt_in(monkeypatch, exporter, spec, exports):
    monkeypatch.setenv("MILES_LENS_SPAN_GROUPS", spec)
    observer = LensObserver()
    observer.start_process(_process(role="rollout_manager", rank=-1))
    observer.span_start(_span(1, "rollout.get", "rollout", 0.0, 0.0, rollout_id=2))
    for kind, ts, turn, detail in [
        ("attempt_start", 1.0, -1, ""),
        ("gen_start", 1.5, 1, ""),
        ("gen_end", 3.0, 1, "128"),
        ("tool_start", 3.0, 1, "search"),
        ("tool_end", 3.5, 1, "search"),
        ("attempt_end", 4.0, -1, "completed"),
    ]:
        event = {"ts": ts, "kind": kind, "sample_index": 9, "group_index": 2, "turn": turn}
        observer.trajectory_event({**event, "weight_version": "5", "detail": detail})

    spans = _finished(observer, exporter)

    names = {"sample.queue", "sample.generate", "sample.tool", "sample.attempt"}
    assert (names <= spans.keys()) is exports
    if exports:
        generate = spans["sample.generate"]
        assert (generate.start_time, generate.end_time) == (1_500_000_000, 3_000_000_000)
        assert generate.attributes["miles.completion_tokens"] == "128"
        assert generate.attributes["miles.rollout_id"] == 2
        assert spans["sample.queue"].end_time == 1_500_000_000
        assert spans["sample.attempt"].attributes["miles.status"] == "completed"


def test_unselected_training_ranks_export_nothing(monkeypatch, exporter):
    monkeypatch.setenv("MILES_LENS_TRAIN_RANKS", "0,8")
    observer = LensObserver()
    observer.start_process(_process(rank=1))
    observer.phase_end("actor_train", 1.0, 2.0)
    assert not observer.wants_device_spans()
    assert _finished(observer, exporter) == {}


@pytest.mark.parametrize(("spec", "wanted"), [("default", False), ("device", True), ("all", True)])
def test_device_spans_follow_the_gpu_group(monkeypatch, exporter, spec, wanted):
    monkeypatch.setenv("MILES_LENS_SPAN_GROUPS", spec)
    observer = LensObserver()
    observer.start_process(_process())
    assert observer.wants_device_spans() is wanted
    observer.span_end(_span(1, "forward", "gpu", 1.0, 1.25, microbatch=0))
    assert ("forward" in _finished(observer, exporter)) is wanted
