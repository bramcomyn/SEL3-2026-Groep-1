from brittle_star_locomotion.config.configuration import Configuration

import jax
import jax.numpy as jnp


class ArmDamage():
    def __init__(self):
        self._config = Configuration().configuration
        self._n_environments = self._config.environment.number_of_environments
        self._n_agents = self._config.environment.number_of_arms
        self._seed = self._config.damage.seed
        self._rng = jax.random.PRNGKey(self._seed)

        self._breakpoint_range = self._config.damage.breakpoint_range

        self._active_arms = jnp.ones((self._n_environments, self._n_agents))     # shape (n_envs, n_agents)
        self._break_points = jax.random.randint(self._rng, (self._n_environments), 3, 15) # shape (n_envs,)

    def break_arms(self, step_idx: int) -> None:
        """Deactivates a random arm in an environment if that environment reached its breakpoint

        :param step_idx: The current step in the environment
        """
        trigger = step_idx == self._break_points # shape (n_envs,)

        self._rng, subkey = jax.random.split(self._rng)
        random_arms = jax.random.randint(subkey, (self._n_environments), 0, self._n_agents) # shape (n_envs,)

        arm_mask = jax.nn.one_hot(random_arms, self._n_agents) # (n_envs, n_arms)

        self._active_arms = jnp.where(          # (n_envs, n_arms)
            trigger[:, None],                   # (n_envs, 1)
            self._active_arms * (1 - arm_mask), # Turn off
            self._active_arms
        )

    def reset(self) -> None:
        """Resets the active arms and breakpoints"""
        self._active_arms = jnp.ones((self._n_environments, self._n_agents))     # shape (n_envs, n_agents)
        self._break_points = jax.random.randint(self._rng, (self._n_environments), 3, 15) # shape (n_envs,)

    def get_active_arms(self) -> jnp.ndarray:
        """Returns the active arms.
        
        :return: The active arms (1 is active, 0 is not active) with shape (n_envs, n_arms)
        """
        return self._active_arms
