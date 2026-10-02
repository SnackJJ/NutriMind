"""Synchronous veRL trainer used by the NutriMind agent baseline."""

from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync


class NutriMindSyncTrainer(PPOTrainerSync):
    def step(self, metrics, timing_raw, *, prefetch_next_batch=True):
        # A synchronous step always needs its current-policy batch, including
        # the terminal step where the base trainer disables future prefetch.
        self._add_batch_to_generate()
        return super().step(metrics, timing_raw, prefetch_next_batch=False)
