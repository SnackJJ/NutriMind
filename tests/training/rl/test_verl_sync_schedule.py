"""The final synchronous step must submit a fresh group, not wait on an empty queue."""

from types import SimpleNamespace

import pytest

pytest.importorskip("verl.trainer.ppo.v1.trainer_sync")
from src.training.rl.verl_trainer import NutriMindSyncTrainer


@pytest.mark.parametrize("last_step", [False, True])
def test_every_sync_step_queues_exactly_one_current_batch(last_step):
    trainer = object.__new__(NutriMindSyncTrainer)
    trainer.config = SimpleNamespace(data=SimpleNamespace(train_batch_size=2))
    trainer.parameter_sync_step = 1
    queued = []
    trainer._add_batch_to_generate = lambda: queued.append("current-policy")

    def consume(metrics, timing, sample_batch_size):
        assert queued == ["current-policy"], "no batch submitted for the last synchronous step"
        queued.pop()
        return SimpleNamespace(keys=["a", "b"], tags=[{}, {}], partition_id="train")

    trainer._step_once = consume
    trainer.step({}, {}, prefetch_next_batch=not last_step)
    assert queued == []
