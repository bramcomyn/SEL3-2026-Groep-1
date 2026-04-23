from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class Transition:
    """Class that represents a transition tuple `(s, a, r, s')` from reinforcement learning."""
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray
