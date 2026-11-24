"""Turn an OpenTelemetry collector's file export into a timeline for ui.perfetto.dev.

    miles-lens-perfetto spans.jsonl -o timeline.json [--summary]

The input is what the collector's ``file`` exporter writes: one OTLP/JSON
``ExportTraceServiceRequest`` per line. The output is a Chrome trace: one
process per miles process (driver, controllers, each training rank), one lane
per span group, with overlapping spans of a group spread over numbered lanes.
``--summary`` prints, per step, how long each group's spans took in total.
"""

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from miles_lens.timers import timer_group

# processes top to bottom in the order a step flows through them
_ROLE_ORDER = ["driver", "inference_controller", "rollout_manager", "trainer_controller", "train", "command"]


@dataclass(frozen=True)
class Span:
    process: tuple[str, int, str, int]  # role, rank, node, pid
    name: str
    group: str
    start_ns: int
    end_ns: int
    attributes: dict[str, Any]


def read_spans(lines: Iterable[str]) -> Iterator[Span]:
    for line in lines:
        if not line.strip():
            continue
        for resource_spans in json.loads(line).get("resourceSpans", []):
            resource = _attributes(resource_spans.get("resource", {}).get("attributes", []))
            process = (
                str(resource.get("nv.dl.role", resource.get("service.name", "?"))),
                int(resource.get("nv.dl.rank", -1)),
                str(resource.get("miles.node", resource.get("host.name", ""))),
                int(resource.get("process.pid", 0)),
            )
            for scope_spans in resource_spans.get("scopeSpans", []):
                for span in scope_spans.get("spans", []):
                    attributes = _attributes(span.get("attributes", []))
                    yield Span(
                        process=process,
                        name=span["name"],
                        group=_group_of(span["name"], attributes),
                        start_ns=int(span["startTimeUnixNano"]),
                        end_ns=int(span["endTimeUnixNano"]),
                        attributes=attributes,
                    )


def chrome_trace(spans: list[Span]) -> dict[str, Any]:
    processes = sorted({span.process for span in spans}, key=_process_order)
    pid_of = {process: pid for pid, process in enumerate(processes, start=1)}
    events: list[dict[str, Any]] = []
    for process, pid in pid_of.items():
        role, rank, node, os_pid = process
        label = f"{role}" + (f" rank {rank}" if rank >= 0 else "") + f" @ {node} (pid {os_pid})"
        events.append({"ph": "M", "name": "process_name", "pid": pid, "args": {"name": label}})
        events.append({"ph": "M", "name": "process_sort_index", "pid": pid, "args": {"sort_index": pid}})

    lanes: dict[tuple, list[int]] = defaultdict(list)  # (process, group) -> end of the last span in each lane
    tid_of: dict[tuple, int] = {}
    for span in sorted(spans, key=lambda s: (s.start_ns, -s.end_ns)):
        lane_ends = lanes[(span.process, span.group)]
        lane = next((i for i, end in enumerate(lane_ends) if end <= span.start_ns), len(lane_ends))
        if lane == len(lane_ends):
            lane_ends.append(span.end_ns)
        else:
            lane_ends[lane] = span.end_ns
        key = (span.process, span.group, lane)
        if key not in tid_of:
            tid_of[key] = len(tid_of) + 1
            pid = pid_of[span.process]
            name = span.group if lane == 0 else f"{span.group} #{lane}"
            events.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": tid_of[key], "args": {"name": name}})
            sort_index = _group_order(span.group) * 1000 + lane
            events.append(
                {
                    "ph": "M",
                    "name": "thread_sort_index",
                    "pid": pid,
                    "tid": tid_of[key],
                    "args": {"sort_index": sort_index},
                }
            )
        events.append(
            {
                "ph": "X",
                "name": span.name,
                "cat": span.group,
                "pid": pid_of[span.process],
                "tid": tid_of[key],
                "ts": span.start_ns / 1000,
                "dur": (span.end_ns - span.start_ns) / 1000,
                "args": span.attributes,
            }
        )
    return {"traceEvents": events, "displayTimeUnit": "ms"}


def step_summary(spans: list[Span]) -> str:
    """Seconds spent per group in each step, summed over spans; parallel work counts once per span."""
    totals: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for span in spans:
        if (rollout_id := span.attributes.get("miles.rollout_id")) is not None and span.group != "sample":
            totals[int(rollout_id)][f"{span.process[0]}:{span.name}"] += (span.end_ns - span.start_ns) / 1e9
    columns = sorted({column for step in totals.values() for column in step})
    rows = [["step", *columns]]
    rows += [[str(step), *(f"{totals[step].get(column, 0.0):.2f}" for column in columns)] for step in sorted(totals)]
    return "\n".join(_transpose_table(rows))


def _transpose_table(rows: list[list[str]]) -> list[str]:
    # one line per span name reads better than one very wide line per step
    columns = list(zip(*rows, strict=True))
    width = max(len(column[0]) for column in columns)
    return [f"{column[0]:<{width}}  " + "  ".join(f"{cell:>8}" for cell in column[1:]) for column in columns]


def _attributes(items: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for item in items:
        value = item.get("value", {})
        for kind in ("stringValue", "intValue", "doubleValue", "boolValue"):
            if kind in value:
                out[item["key"]] = int(value[kind]) if kind == "intValue" else value[kind]
                break
    return out


def _group_of(name: str, attributes: dict[str, Any]) -> str:
    # spans from before the plugin recorded its group fall back to the timer table
    return str(attributes.get("miles.group") or timer_group(name))


def _process_order(process: tuple[str, int, str, int]) -> tuple:
    role, rank, node, pid = process
    return (_ROLE_ORDER.index(role) if role in _ROLE_ORDER else len(_ROLE_ORDER), rank, node, pid)


_GROUP_ORDER = ["step", "rollout", "sample", "dispatch", "train", "gpu", "weight_sync", "offload", "checkpoint"]


def _group_order(group: str) -> int:
    return _GROUP_ORDER.index(group) if group in _GROUP_ORDER else len(_GROUP_ORDER)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, nargs="+", help="collector file-exporter output (JSON lines)")
    parser.add_argument("-o", "--output", type=Path, default=Path("timeline.json"))
    parser.add_argument("--summary", action="store_true", help="print per-step seconds by span")
    args = parser.parse_args()

    spans = [span for path in args.input for span in read_spans(path.read_text().splitlines())]
    args.output.write_text(json.dumps(chrome_trace(spans)))
    print(f"{len(spans)} spans from {len({s.process for s in spans})} processes -> {args.output}")
    if args.summary:
        print(step_summary(spans))


if __name__ == "__main__":
    main()
