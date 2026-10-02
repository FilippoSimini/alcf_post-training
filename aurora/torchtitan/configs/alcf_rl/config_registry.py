"""Extra GRPO configs for the alphabet_sort task, kept outside the torchtitan checkout.

torchtitan's --module accepts a fully qualified module path, so additional
configs can live here instead of being patched into the installed source tree.
Select one with:

    --module alcf_rl --config rl_grpo_lora_qwen3_4b

run_grpo.sh puts this directory on PYTHONPATH.
"""

# Re-export the upstream configs so --module alcf_rl can reach them too.
from torchtitan.experiments.rl.examples.alphabet_sort.config_registry import *  # noqa: F401,F403
from torchtitan.experiments.rl.examples.alphabet_sort.config_registry import (
    _qwen3_rl_model_registry,
    rl_grpo_full_qwen3_0_6b_flex,
    rl_grpo_lora_qwen3_0_6b,
)

from torchtitan.components.lora import LoRAConverter
from torchtitan.components.optimizer import default_adamw
from torchtitan.experiments.rl.controller import Controller
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor


def _lora_converters():
    return [LoRAConverter.Config(rank=32, alpha=64.0, target_modules=["wqkv", "wo"])]


# Every generator metric is emitted with both Mean and Max aggregators, but the
# default console filter prints Mean only. Max is the straggler request, and the
# step waits on the slowest rollout rather than the average one. Also surface
# time_to_first_token and prefill_time, which are computed and never printed, so
# prefill can be separated from decode.
_CONSOLE_KEYS_TRAIN = [
    "perf/",
    "generator/inflight_requests_at_completion/max",
    "generator/inter_token_latency_ms/mean",
    "generator/inter_token_latency_ms/max",
    "generator/queue_time_ms/mean",
    "generator/queue_time_ms/max",
    "generator/decode_time_ms/mean",
    "generator/decode_time_ms/max",
    "generator/time_to_first_token_ms/mean",
    "generator/time_to_first_token_ms/max",
    "generator/prefill_time_ms/mean",
    "generator/prefill_time_ms/max",
    "generator/num_cached_tokens/mean",
    # Batcher packing. Computed on every step by Batcher._packing_metrics and
    # otherwise discarded: padding_frac and num_microbatches are what show
    # whether local_batch_size and dp_degree are matched. At high dp_degree the
    # microbatch grid is filled out with pad-only rows, and padding_frac is the
    # direct measure of that waste.
    "train_batch/",
    "loss/mean",
    "rollout_reward/_mean",
    # A group whose siblings all score the same carries no GRPO signal and is
    # dropped. If EVERY group drops, the batcher never fills, the trainer never
    # steps, and nothing is logged at all -- the 8-node silent hang. These two
    # make that visible as it develops rather than as an absence.
    "rollout_reward/group_zero_std_frac",
    "training_sample_builder/num_groups_dropped_zero_std",
    "trainer/entropy/mean",
    "trainer/grad_norm/mean",
    "trainer/lr",
    "bit_wise/logprob_diff/max",
]


def _with_max_metrics(cfg: Controller.Config) -> Controller.Config:
    cfg.metrics = MetricsProcessor.Config(
        enable_wandb=False, console_log_keys_train=list(_CONSOLE_KEYS_TRAIN)
    )
    return cfg


def rl_grpo_lora_qwen3_0_6b_lr2e5() -> Controller.Config:
    """Qwen3-0.6B LoRA at lr=2e-5, with Max generator metrics on the console.

    The shipped config uses 2e-6, carried over from the non-LoRA parallelism
    sweep. At that rate the adapters barely move and rollout reward drifts
    down; every other LoRA config in the upstream registry uses 1e-4 or 2e-5.
    """
    cfg = rl_grpo_lora_qwen3_0_6b()
    cfg.trainer.optimizer = default_adamw(lr=2e-5)
    return _with_max_metrics(cfg)


def rl_grpo_full_qwen3_0_6b() -> Controller.Config:
    """Upstream's full-parameter twin of the LoRA config, logging widened only.

    Hyperparameters are untouched — including the shipped `lr=2e-6`. The
    `lr=2e-5` finding is LoRA-specific: adapters start at zero and need a large
    rate, whereas full-parameter training starts from pretrained weights and
    normally wants the smaller one. Only `metrics` differs, so the comparison
    against `rl_grpo_lora_qwen3_0_6b_lr2e5` is like-for-like on everything the
    trainer actually does.

    Removing LoRA also removes two constraints: dp_shard no longer has to
    divide 32, and the adapters' missing sharding config no longer blocks
    `spmd_backend=spmd_types`.
    """
    return _with_max_metrics(rl_grpo_full_qwen3_0_6b_flex())


def rl_grpo_lora_qwen3_4b() -> Controller.Config:
    """Qwen3-4B LoRA, for the multi-node scaling sweep.

    The parallelism degrees inherited from the 0.6B config are placeholders:
    multinode_launcher.py derives dp_shard from the node count and overrides
    them.
    """
    cfg = rl_grpo_lora_qwen3_0_6b()
    cfg.model_spec = _qwen3_rl_model_registry(
        "4B", attn_backend="flex", converters=_lora_converters()
    )
    cfg.trainer.optimizer = default_adamw(lr=2e-5)
    return _with_max_metrics(cfg)
