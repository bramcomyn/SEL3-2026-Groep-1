from typing import Callable

import jax
import jax.numpy as jnp
import optax
from brittle_star_locomotion.environment.environment import Environment
from brittle_star_locomotion.nn.q_network import QNetwork
from brittle_star_locomotion.replay_buffer.replay_buffer import ReplayBuffer
from flax import nnx


class IQL:
    def __init__(self, optimizer: optax.GradientTransformationExtraArgs, loss_fn: Callable, n_agents: int, env: Environment):
        self.loss_fn = loss_fn
        self.n_agents = n_agents
        self.env = env
        self.observation_size = self.env.get_observation_size()

        self.rngs = nnx.Rngs(0)

        self.value_network = QNetwork(self.observation_size, 5, rngs=self.rngs, hidden_size=5)
        # self.target_network = QNetwork(self.observation_size, 5, rngs=self.rngs, hidden_size=5)

        self.replay_buffers = [ReplayBuffer(self.observation_size, 5) for _ in range(n_agents)]

        self.optimizer = nnx.Optimizer(self.value_network, optimizer, wrt=nnx.Param)

    def train(self, **kwargs):
        epochs = kwargs.get("epochs", 10)
        epsilon = kwargs.get("epsilon", 0.4)
        discount = kwargs.get("discount", 0.99)
        batch_size = kwargs.get("batch_size", 4)

        for _ in range(epochs):
            observations = self.env.get_observations()  # n_agents x observation_size
            actions = jnp.zeros(self.n_agents, dtype=jnp.int8)  # n_agents

            for agent in range(self.n_agents):
                if self.rngs.uniform(minval=0.0, maxval=1.0) < epsilon:
                    action = actions.at[agent].set(self.rngs.randint((), minval=0, maxval=5))
                else:
                    action = jnp.argmax(self.value_network(observations[agent]))
                actions = actions.at[agent].set(action)

            _, reward, terminated, truncated = self.env.step(actions)  # 1, 1, 1
            done = jnp.any(terminated) or jnp.any(truncated)
            next_observations = self.env.get_observations()  # n_agents x observation_size

            print(f"reward: {reward}")

            for agent in range(self.n_agents):
                # Store transition in replay buffer
                self.replay_buffers[agent].add(
                    observations[agent],  # observation_size
                    actions[agent],
                    reward,
                    next_observations[agent],  # observation_size
                    done,
                )

                # Sample mini-batch from replay buffer
                mini_batch = self.replay_buffers[agent].sample(batch_size)  # batch_size x observation_size

                # Compute target Q-values using target network
                if done:
                    target = reward  # TODO: needs to be reward from replay buffer
                else:
                    target = reward + discount * jnp.max(self.value_network(mini_batch), axis=0)

                # Compute loss
                loss, grads = nnx.value_and_grad(self.loss_fn)(target, self.value_network)

                # Call optimizer
                self.optimizer.update(self.value_network, grads)

                # Update target network param (interval)
                pass


@nnx.jit
def train_step(model, optimizer, x, y) -> jax.Array:
    def loss_fn(model: QNetwork):
        y_pred = model(x)
        return jnp.mean((y_pred - y) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)

    return loss


if __name__ == "__main__":
    print("Training Q-Network...")
    model = QNetwork(5, 5, rngs=nnx.Rngs(0), hidden_size=5)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    rngs = nnx.Rngs(0)
    x, y = jnp.ones((5,)), jnp.ones((5,))

    loss = train_step(model, optimizer, x, y)
    while loss > 1e-6:
        print(f"Loss {loss}")
        loss = train_step(model, optimizer, x, y)
