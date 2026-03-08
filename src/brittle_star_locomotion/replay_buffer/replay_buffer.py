from functools import partial

import jax
import jax.numpy as jnp


class ReplayBuffer:
    def __init__(
        self,
        size_observation: int,
        size_action: int,
        max_size: int = 1000000,
        rng=jax.random.PRNGKey(0)
    ) -> None:
        self.size_observation = size_observation
        self.size_action = size_action
        self.max_size = max_size
        self.replay_buffer = jnp.zeros((max_size, size_observation * 2 + size_action + 2))
        self.size = 0
        self.rng = rng

    def add(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
        reward: float,
        next_observation: jnp.ndarray,
        done: bool
    ) -> None:
        """Add an experience to the replay buffer.

        :param observation: The current observation of the environment.
        :type observation: jnp.ndarray

        :param action: The action taken by the agent.
        :type action: jnp.ndarray

        :param reward: The reward received for taking the action.
        :type reward: float

        :param next_observation: The next observation of the environment.
        :type next_observation: jnp.ndarray

        :param done: A boolean indicating if the episode is done.
        :type done: bool
        """
        new_entry = jnp.concatenate([observation,
            action,
            jnp.array([reward]),
            next_observation,
            jnp.array([done])
            ])
        self.replay_buffer.at[self.size].set(new_entry)
        self.size = (self.size + 1) % self.max_size

    @partial(jax.jit, static_argnums=(0,1))
    def sample(
        self,
        batch_size: int
    ) -> jnp.ndarray:
        """Sample a batch of experiences from the replay buffer.

        :param batch_size: The number of samples to return.
        :type batch_size: int

        :return: A batch of experiences sampled from the replay buffer.
        :rtype: jnp.ndarray
        """
        random_indices = jax.random.choice(
            self.rng,
            self.size,
            shape=(batch_size,),
            replace=False
        )
        return self.replay_buffer[random_indices]
