"""veRL trainers used by the NutriMind agent baseline."""

from omegaconf import OmegaConf

from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync


class NutriMindSyncTrainer(PPOTrainerSync):
    def step(self, metrics, timing_raw, *, prefetch_next_batch=True):
        # A synchronous step always needs its current-policy batch, including
        # the terminal step where the base trainer disables future prefetch.
        self._add_batch_to_generate()
        return super().step(metrics, timing_raw, prefetch_next_batch=False)


class NutriMindSeparateAsyncTrainer(PPOTrainerSeparateAsync):
    # Warmup batches plus one prefetch per non-terminal step already cover every
    # step, so the upstream schedule needs no adapter here.

    def _load_checkpoint(self):
        super()._load_checkpoint()
        # PPOTrainer._setup ends here after launching the hybrid replica on the
        # trainer GPU; PPOTrainerSeparateAsync._setup launches the standalone
        # replica next from the same rollout config.
        utilization = OmegaConf.select(self.config, "trainer.v1.separate_async.standalone_gpu_memory_utilization")
        if utilization is not None:
            self.config.actor_rollout_ref.rollout.gpu_memory_utilization = utilization


TRAINERS = {"sync": NutriMindSyncTrainer, "separate_async": NutriMindSeparateAsyncTrainer}


def trainer_class(config):
    mode = config.trainer.v1.trainer_mode
    if mode not in TRAINERS:
        raise ValueError(f"unsupported trainer.v1.trainer_mode: {mode}")
    return TRAINERS[mode]
