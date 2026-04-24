from brittle_star_locomotion.logger.logger import Logger
from brittle_star_locomotion.config.configuration import Configuration
from brittle_star_locomotion.optimizer.transition import Transition

import jax.numpy as jnp


class TrainingMetrics():
    def __init__(self):
        """
        """
        self._config = Configuration().configuration
        self._logger = Logger()
        self._logger.initialize_wandb(
            project="brittle-star-locomotion", 
            config=self._config, 
            enabled=self._config.logging.use_wandb
        )

        self.total_train_steps = 0
        self.total_environment_steps = 0
        self.episode_index = 0
        self.epsilon = 0.0
        self.episode_index = 0
        self.episode_steps = 0
        self.episode_reward_sum = jnp.zeros((self._config.environment.number_of_environments,), dtype=jnp.float32)
        self.episode_loss_sum = jnp.zeros((self._config.environment.number_of_environments,), dtype=jnp.float32)
        self.episode_terminated_count = 0
        self.episode_truncated_count = 0


    def new_training(self) -> None:
        self.total_train_steps = 0
        self.total_environment_steps = 0

        self._logger.info(
            f"Starting IQL training for {self._config.rl.n_episodes} episodes "
            f"across {self._config.environment.number_of_environments} environments"
        )

    def end_training(self) -> None:
        self._logger.info(
            f"Finished IQL training after {self.total_environment_steps} environment steps and "
            f"{self.total_train_steps} optimization steps"
        )

    def new_episode(self, epsilon: float) -> None:
        """
        """
        self.epsilon = epsilon
        self.episode_index += 1
        self.episode_steps = 0
        self.episode_reward_sum = jnp.zeros((self._config.environment.number_of_environments,), dtype=jnp.float32)
        self.episode_loss_sum = jnp.zeros((self._config.environment.number_of_environments,), dtype=jnp.float32)
        self.episode_terminated_count = 0
        self.episode_truncated_count = 0

        previous_epsilon = self.epsilon
        self._logger.info(
            f"Episode {self.episode_index}/{self._config.rl.n_episodes} started: "
            f"epsilon {previous_epsilon:.4f} -> {epsilon:.4f}"
            )

    def new_episode_step(
        self,
        transition: Transition,
        losses: jnp.ndarray
    ) -> None:
        """
        """
        self.episode_steps += 1
        self.total_environment_steps += 1
        self.episode_reward_sum += transition.rewards
        self.episode_loss_sum += losses
        self.episode_truncated_count = int(jnp.sum(transition.truncated))
        self.episode_terminated_count = int(jnp.sum(transition.terminated))

    def end_episode(self) -> None:
        """
        """
        mean_return = float(jnp.mean(self.episode_reward_sum))
        mean_loss = float(jnp.mean(self.episode_loss_sum))

        self._logger.info(
            f"Episode {self.episode_index}/{self._config.rl.n_episodes} finished: "
            f"amount_steps={self.episode_steps}, "
            f"mean_return={mean_return:.4f}, "
            f"mean_loss={mean_loss:.4f}, "
            f"terminated={self.episode_terminated_count}, "
            f"truncated={self.episode_truncated_count}, "
            f"total_train_steps={self.total_train_steps}"
        )
        self._logger.log_metrics({
            "global/epsilon": self.epsilon,
            "global/terminated":self.episode_terminated_count,
            "global/truncated": self.episode_truncated_count,
            "optimizer/mean_return": mean_return,
            "optimizer/mean_loss": mean_loss
        })
