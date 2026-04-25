from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class Transition:
    """Class that represents a transition tuple `(s, a, r, s')` from reinforcement learning."""
    observations: jnp.ndarray       # shape (n_environments, n_agents, observation_size)
    actions: jnp.ndarray            # shape (n_environments, n_agents)
    rewards: jnp.ndarray            # shape (n_environments,)
    next_observations: jnp.ndarray  # shape (n_environments, n_agents, observation_size)
    terminated: jnp.ndarray         # shape (n_environments,)
    truncated: jnp.ndarray          # shape (n_environments,)
