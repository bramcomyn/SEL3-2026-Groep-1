from brittle_star_locomotion.environment.environment import Environment
from brittle_star_locomotion.config.configuration import Configuration

import jax
import jax.numpy as jnp


class RandomTargetEnvironment(Environment):
    """Environment where the target position is randomly generated at the start of each episode."""
    def __init__(self, *args, **kwargs):
        config = Configuration().configuration
        self._n_environments = config.environment.number_of_environments
        self._target_distance = config.environment.target_distance
        self._seed = config.environment.seed

        self.target_position = self._random_target_position()

        super().__init__(*args, **kwargs)

    def _random_target_position(self) -> jnp.ndarray:
        """Generates random target positions
        
        :returns: Random target positions of shape (n_environments,)
        """
        rng = jax.random.PRNGKey(self._seed)

        angle = jax.random.uniform(key=rng, shape=(self._n_environments,), minval=0, maxval=jnp.pi * 2) # shape (n_environments,)
        radius = self._target_distance
        random_position = jnp.stack(
            [radius * jnp.cos(angle), radius * jnp.sin(angle), jnp.repeat(0.05, self._n_environments)],
            axis=-1
        )

        return random_position # shape (n_environments,3)

    def _reset_all_envs(self, sub_rngs: jnp.ndarray):
        """Reset all environments while avoiding vmapped-reset instability for single-env runs."""
        # if self.number_of_environments == 1:
        #     single_env_state = self.jit_env_reset_single(sub_rngs[0], self.target_position)
        #     return jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], single_env_state)
        
        target_position_for_all_envs = jnp.broadcast_to(self.target_position, (self.number_of_environments, 3)) # shape: (envs, 3)
        return self.jit_env_reset(sub_rngs, target_position_for_all_envs)
