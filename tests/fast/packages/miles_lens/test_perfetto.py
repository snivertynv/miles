"""The collector's file export becomes a Perfetto timeline: a process per miles process, valid lanes per group."""

import json

from miles_lens.perfetto import chrome_trace, read_spans, step_summary


def _line(role, rank, spans):
    def value(v):
        return {"intValue": str(v)} if isinstance(v, int) else {"stringValue": v}

    return json.dumps(
        {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "nv.dl.role", "value": value(role)},
                            {"key": "nv.dl.rank", "value": value(rank)},
                            {"key": "miles.node", "value": value("10.0.0.1")},
                            {"key": "process.pid", "value": value(100 + rank)},
                        ]
                    },
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "name": name,
                                    "startTimeUnixNano": str(int(t0 * 1e9)),
                                    "endTimeUnixNano": str(int(t1 * 1e9)),
                                    "attributes": [{"key": k, "value": value(v)} for k, v in attrs.items()],
                                }
                                for name, t0, t1, attrs in spans
                            ]
                        }
                    ],
                }
            ]
        }
    )


LINES = [
    _line("train", 1, [("actor_train", 4.0, 8.0, {"miles.group": "train", "miles.rollout_id": 0})]),
    _line(
        "rollout_manager",
        -1,
        [
            ("sample.generate", 0.0, 3.0, {"miles.group": "sample", "miles.rollout_id": 0}),
            ("sample.generate", 1.0, 2.0, {"miles.group": "sample", "miles.rollout_id": 0}),
            ("sample.generate", 3.5, 4.0, {"miles.group": "sample", "miles.rollout_id": 0}),
        ],
    ),
    _line("driver", -1, [("driver.step", 0.0, 10.0, {"miles.group": "step", "miles.rollout_id": 0})]),
]


def test_processes_follow_the_step_and_overlaps_get_their_own_lane():
    events = chrome_trace(list(read_spans(LINES)))["traceEvents"]

    processes = sorted((e["args"]["sort_index"], e["pid"]) for e in events if e["name"] == "process_sort_index")
    names = {e["pid"]: e["args"]["name"] for e in events if e["name"] == "process_name"}
    assert [names[pid].split(" @")[0] for _, pid in processes] == ["driver", "rollout_manager", "train rank 1"]

    lane_names = {e["tid"]: e["args"]["name"] for e in events if e["name"] == "thread_name"}
    generate = [e for e in events if e.get("name") == "sample.generate"]
    assert sorted(lane_names[e["tid"]] for e in generate) == ["sample", "sample", "sample #1"]
    for tid in {e["tid"] for e in generate}:
        on_lane = sorted((e["ts"], e["ts"] + e["dur"]) for e in generate if e["tid"] == tid)
        assert all(end <= next_start for (_, end), (next_start, _) in zip(on_lane, on_lane[1:], strict=False))
    assert next(e for e in events if e.get("name") == "actor_train")["dur"] == 4_000_000


def test_summary_adds_up_each_span_per_step_and_leaves_samples_out():
    summary = step_summary(list(read_spans(LINES)))
    rows = {line.split()[0]: line.split()[1:] for line in summary.splitlines()}
    assert rows["step"] == ["0"]
    assert rows["train:actor_train"] == ["4.00"] and rows["driver:driver.step"] == ["10.00"]
    assert not any("sample" in key for key in rows)
