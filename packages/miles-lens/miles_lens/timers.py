"""Which span group each miles timer belongs to; no nemo-lens import, so the converter runs anywhere."""

# miles timer names (miles/utils/timer.py) by group; a ``<model>/`` prefix is ignored
_TIMER_GROUPS = {
    "rollout": "rollout",
    "eval_rollout": "eval",
    "data_preprocess": "dispatch",
    "train_wait": "train",
    "train": "train",
    "actor_train": "train",
    "critic_train": "train",
    "log_probs": "train",
    "ref_log_probs": "train",
    "teacher_log_probs": "train",
    "ref_model_update": "train",
    "sleep": "offload",
    "wake_up": "offload",
    "offload_grad_buffer": "offload",
    "save_model": "checkpoint",
    "export_hf": "checkpoint",
    "update_weights": "weight_sync",
    "update_weights_implementation": "weight_sync",
    "finalize_and_resume_engines": "weight_sync",
    "ici_send": "weight_sync",
    "ici_engine_load": "weight_sync",
}


def timer_group(name: str) -> str:
    return _TIMER_GROUPS.get(name.rpartition("/")[2], "other")
