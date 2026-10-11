"""CPU checks for the one-step off-policy separate-async profile.

Run in the GRPO environment with MIMO_VERL_ROOT on PYTHONPATH.
"""

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("verl.trainer.ppo.v1.trainer_separate_async")
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync

from src.training.rl.verl_agent_loop import _merge_engine_fields
from src.training.rl.verl_trainer import (
    NutriMindSeparateAsyncTrainer, NutriMindSyncTrainer, trainer_class)

CONFIGS = Path(__file__).resolve().parents[3] / "configs"
# Everything async needs; any other difference from the synchronous run is a hyperparameter change.
ASYNC_ONLY = ("algorithm.rollout_correction.", "actor_rollout_ref.rollout.nnodes",
              "actor_rollout_ref.rollout.n_gpus_per_node", "actor_rollout_ref.rollout.gpu_memory_utilization",
              "actor_rollout_ref.rollout.checkpoint_engine.", "trainer.v1.", "trainer.experiment_name",
              "trainer.default_local_dir", "trainer.rollout_data_dir",
              # Same weights with text-only key names, which the standalone replica's disk load needs.
              "actor_rollout_ref.model.path",
              # Run length and checkpoint/validation schedule, not algorithm settings.
              "trainer.total_epochs", "trainer.total_training_steps", "trainer.save_freq", "trainer.test_freq",
              "trainer.max_actor_ckpt_to_keep")


def _compose(name):
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        return compose(config_name=name, overrides=["hydra.searchpath=[pkg://verl.trainer.config]"])


def _flatten(node, prefix=""):
    if isinstance(node, dict):
        return {k: v for key, value in node.items() for k, v in _flatten(value, f"{prefix}{key}.").items()}
    return {prefix[:-1]: node}


def test_async_profile_changes_only_async_settings():
    sync = _flatten(OmegaConf.to_container(_compose("grpo_v2_a800_bs16")))
    async_ = _flatten(OmegaConf.to_container(_compose("grpo_v2_4090x2_async")))
    changed = {k for k in sync.keys() | async_.keys() if sync.get(k, "<missing>") != async_.get(k, "<missing>")}
    assert changed and all(k.startswith(ASYNC_ONLY) for k in changed), sorted(changed)


def test_async_profile_is_one_step_off_policy_with_truncated_is():
    config = _compose("grpo_v2_4090x2_async")
    v1 = config.trainer.v1
    assert trainer_class(config) is NutriMindSeparateAsyncTrainer
    assert v1.separate_async.parameter_sync_step == 1 and v1.separate_async.num_warmup_batches == 1
    # A prompt submitted at step s is sampled at step s + 1 at the latest (span 2).
    assert v1.sampler.max_off_policy_threshold == 2
    correction = config.algorithm.rollout_correction
    assert correction.rollout_is == "token" and not correction.bypass_mode
    assert config.actor_rollout_ref.rollout.calculate_log_probs


def test_upstream_separate_async_accepts_profile():
    trainer = PPOTrainerSeparateAsync(_compose("grpo_v2_4090x2_async"))
    assert trainer.trainer_mode == "separate_async" and trainer.parameter_sync_step == 1
    assert type(trainer.replay_buffer).__name__ == "ReplayBufferAsync"


def test_sync_profiles_keep_sync_trainer():
    for name in ("grpo_v2_4090", "grpo_v2_a800_bs16"):
        assert trainer_class(_compose(name)) is NutriMindSyncTrainer
    with pytest.raises(ValueError, match="colocate_async"):
        trainer_class(OmegaConf.create({"trainer": {"v1": {"trainer_mode": "colocate_async"}}}))


@pytest.mark.parametrize("steps", [1, 3])
def test_warmup_and_prefetch_cover_every_async_step(steps):
    trainer = object.__new__(NutriMindSeparateAsyncTrainer)
    trainer.config = OmegaConf.create({"skip": {"rollout_tq": {"enable": False}}, "data": {"train_batch_size": 2},
                                       "trainer": {"v1": {"separate_async": {"num_warmup_batches": 1}}}})
    trainer.parameter_sync_step = 1
    queued = []
    trainer._add_batch_to_generate = lambda: queued.append("batch")

    def consume(metrics, timing, sample_batch_size):
        assert queued, "no batch submitted for this step"
        queued.pop()
        return SimpleNamespace(keys=["a", "b"], tags=[{}, {}], partition_id="train")

    trainer._step_once = consume
    trainer.on_train_begin()
    for step in range(1, steps + 1):
        trainer.step({}, {}, prefetch_next_batch=step < steps)
    assert queued == []


def test_standalone_replica_gets_its_own_memory_budget(monkeypatch):
    # The override relies on the pinned setup order: hybrid launch, checkpoint load, standalone launch.
    base = inspect.getsource(PPOTrainer._setup)
    assert base.index("LLMServerManager.create") < base.index("self._load_checkpoint()")
    separate = inspect.getsource(PPOTrainerSeparateAsync._setup)
    assert separate.index("super()._setup()") < separate.index("LLMServerManager.create")
    monkeypatch.setattr(PPOTrainer, "_load_checkpoint", lambda self: None)
    trainer = object.__new__(NutriMindSeparateAsyncTrainer)
    trainer.config = _compose("grpo_v2_4090x2_async")
    assert trainer.config.actor_rollout_ref.rollout.gpu_memory_utilization == 0.45
    trainer._load_checkpoint()
    assert trainer.config.actor_rollout_ref.rollout.gpu_memory_utilization == 0.75


def test_trajectory_version_range_spans_all_turns():
    fields = {}
    for turn in ({"global_steps": 4, "min_global_steps": 4, "max_global_steps": 4},
                 {"global_steps": 5, "min_global_steps": 4, "max_global_steps": 5},
                 {"global_steps": 5, "min_global_steps": 5, "max_global_steps": 5}, None):
        fields = _merge_engine_fields(fields, turn)
    assert fields == {"global_steps": 5, "min_global_steps": 4, "max_global_steps": 5}
    # Version 0 is a real policy version, and a turn without versions keeps the known range.
    assert _merge_engine_fields({"min_global_steps": 0, "max_global_steps": 0},
                                {"min_global_steps": None, "max_global_steps": 1}) == \
        {"min_global_steps": 0, "max_global_steps": 1}
