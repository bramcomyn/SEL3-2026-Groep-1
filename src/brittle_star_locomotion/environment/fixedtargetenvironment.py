import jax
import jax.numpy as jnp

from brittle_star_locomotion.config.configuration import Configuration
from brittle_star_locomotion.environment.environment import Environment

class FixedTargetEnvironment(Environment):
    """Environment where the target position is fixed at the start of each episode."""
    def __init__(self, *args, **kwargs):
        config = Configuration().configuration
        self.target_position = jnp.array(config.environment.target_position) # shape (3,)

        super().__init__(*args, **kwargs)

    def _reset_all_envs(self, sub_rngs: jnp.ndarray):
        """Reset all environments while avoiding vmapped-reset instability for single-env runs."""
        target_position_for_all_envs = jnp.broadcast_to(self.target_position, (self.number_of_environments, 3)) # shape: (envs, 3)
        return self.jit_env_reset(sub_rngs, target_position_for_all_envs)
