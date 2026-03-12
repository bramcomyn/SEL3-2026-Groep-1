from typing import Callable

import jax.numpy as jnp
import optax
from brittle_star_locomotion.environment.environment import Environment
from brittle_star_locomotion.nn.q_network import QNetwork
from cpprb import ReplayBuffer
from flax import nnx


class IQL:
    def __init__(self, optimizer: optax.GradientTransformationExtraArgs, loss_fn: Callable, n_agents: int, env: Environment, **kwargs):
        self.replay_buffer_size = kwargs.get("replay_buffer_size", 5_000)

        self.loss_fn = loss_fn
        self.n_agents = n_agents
        self.env = env
        self.observation_size = self.env.get_observation_size()

        self.rngs = nnx.Rngs(0)

        self.value_network = QNetwork(self.observation_size, 5, rngs=self.rngs, hidden_size=5)
        # self.target_network = QNetwork(self.observation_size, 5, rngs=self.rngs, hidden_size=5)

        self.replay_buffers = [
            ReplayBuffer(
                self.replay_buffer_size,
                env_dict={"obs": {"shape": self.observation_size}, "act": {}, "rew": {}, "next_obs": {"shape": self.observation_size}, "done": {}},
            )
            for _ in range(n_agents)
        ]

        self.optimizer = nnx.Optimizer(self.value_network, optimizer, wrt=nnx.Param)

    def train(self, **kwargs):
        epochs = kwargs.get("epochs", 10)
        epsilon = kwargs.get("epsilon", 0.4)
        discount = kwargs.get("discount", 0.99)
        batch_size = kwargs.get("batch_size", 4)

        for _ in range(epochs):
            observations = self.env.get_observations()  # n_agents x observation_size
            actions = jnp.zeros(self.n_agents, dtype=jnp.int32)  # n_agents

            for agent in range(self.n_agents):
                if self.rngs.uniform(minval=0.0, maxval=1.0) < epsilon:
                    action = self.rngs.randint((), minval=0, maxval=5)
                else:
                    action = jnp.argmax(self.value_network(observations[agent]))
                actions = actions.at[agent].set(action)

            _, reward, terminated, truncated = self.env.step(actions)  # 1, 1, 1
            done = jnp.logical_or(jnp.any(terminated), jnp.any(truncated))
            next_observations = self.env.get_observations()  # n_agents x observation_size

            print(f"reward: {reward}")

            for agent in range(self.n_agents):
                self.replay_buffers[agent].add(
                    obs=observations[agent],  # observation_size
                    act=actions[agent],
                    rew=reward,
                    next_obs=next_observations[agent],  # observation_size
                    done=done,
                )

                mini_batch = self.replay_buffers[agent].sample(batch_size)

                if done:
                    target = mini_batch["rew"]
                else:
                    target = mini_batch["rew"].squeeze() + discount * jnp.max(self.value_network(mini_batch["next_obs"]), axis=1)

                def loss_fn(model, obs, actions, targets):
                    q_values = model(obs)  # forward pass
                    actions = mini_batch["act"].astype(jnp.int32)  # batch x 1
                    q_selected = jnp.take_along_axis(q_values, actions, axis=1).squeeze()
                    loss = jnp.mean((q_selected - targets) ** 2)
                    return loss

                loss, grads = nnx.value_and_grad(loss_fn)(self.value_network, mini_batch["obs"], mini_batch["act"], target)

                print(f"loss: {loss}")

                self.optimizer.update(self.value_network, grads)

                # TODO: Update target network param (interval)
                pass
