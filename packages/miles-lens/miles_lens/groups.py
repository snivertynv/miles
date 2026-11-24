"""The span groups miles emits, registered with nemo-lens on import.

Groups are miles span categories. ``default`` covers every stage of a step;
``sample`` (one span per generation, tool call and attempt) and ``gpu`` (device
time per microbatch, which adds a synchronize per training call) are opt-in.
"""

from nemo.lens import SpanRegistry

STAGE_GROUPS = frozenset(
    {"step", "init", "rollout", "dispatch", "train", "weight_sync", "checkpoint", "offload", "engine", "eval", "other"}
)
SAMPLE_GROUP = "sample"
GPU_GROUP = "gpu"

SpanRegistry.register(
    "miles",
    groups=STAGE_GROUPS | {SAMPLE_GROUP, GPU_GROUP},
    presets={
        "default": STAGE_GROUPS,
        "per_sample": STAGE_GROUPS | {SAMPLE_GROUP},
        "device": STAGE_GROUPS | {GPU_GROUP},
    },
)
