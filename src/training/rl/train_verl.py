"""MiMo veRL entry point with NutriMind's synchronous scheduling adapter."""

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.logging_utils import configure_verl_logging


@ray.remote
class NutriMindTaskRunner:
    def run(self, config):
        import transfer_queue as tq
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ
        from src.training.rl.verl_trainer import NutriMindSyncTrainer

        configure_verl_logging()
        OmegaConf.resolve(config)
        tq.init(config.transfer_queue)
        trainer = None
        succeeded = False
        try:
            trainer = NutriMindSyncTrainer(config=config)
            trainer.init()
            manager = AgentLoopManagerTQ.create(
                config=config,
                llm_client=trainer.get_llm_client(),
                teacher_client=trainer.get_teacher_client(),
                reward_loop_worker_handles=trainer.get_reward_handles(),
            )
            trainer.fit(manager)
            succeeded = True
        finally:
            try:
                tracking = getattr(trainer, "logger", None)
                if tracking is not None:
                    tracking.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config):
    config.transfer_queue.enable = True
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    run_ppo(config, task_runner_class=NutriMindTaskRunner)


if __name__ == "__main__":
    main()
