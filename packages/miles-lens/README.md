# miles-lens

A `miles.observers` plugin that exports the stages of a miles run as
[nemo-lens](https://github.com/NVIDIA-NeMo/Lens) (OpenTelemetry) spans, so one
timeline shows how rollout, dispatch, training, device work, weight sync,
checkpointing and offload line up across the driver, the controllers and every
training rank.

## Use

```bash
pip install ./packages/miles-lens
export MILES_LENS_ENABLED=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://<collector>:4317
```

The driver forwards the `MILES_LENS_*`, `NEMO_LENS_*` and `OTEL_*` settings to
every worker it launches (except variables whose names suggest credentials, such
as `OTEL_EXPORTER_OTLP_HEADERS`). It also forwards a run id and
`ENABLE_SINGLE_TRACEID=1`, so all processes export into one trace.

| Variable | Default | Meaning |
|---|---|---|
| `MILES_LENS_ENABLED` | off | turn the plugin on |
| `MILES_LENS_SPAN_GROUPS` | `default` | `default` (every stage), `per_sample` (adds a span per generation, tool call and attempt), `device` (adds device-timed forward/backward/optimizer spans, at the cost of a synchronize per training call), `all` |
| `MILES_LENS_TRAIN_RANKS` | `all` | training ranks that export, as a comma-separated list |

Other nemo-lens settings fall back to their `NEMO_LENS_*` names.

## Viewing

Any OTLP backend works. Without one, run an OpenTelemetry collector with the
`file` exporter and convert its output into a trace that ui.perfetto.dev opens:

```bash
miles-lens-perfetto spans.jsonl -o timeline.json --summary
```

The trace has one process per miles process and one lane per span group.
`--summary` prints the seconds each span took in every step.
