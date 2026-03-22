import jax
import jax.numpy as jnp
import optax
import logging
import functools
from flax import nnx
from cpprb import ReplayBuffer

from brittle_star_locomotion.environment import Environment
from brittle_star_locomotion.neural.qnetwork import QNetwork
from brittle_star_locomotion.neural.checkpoint import save_checkpoint
from wandb import agent

logger = logging.getLogger(__name__)


class IndependentQLearning:
    def __init__(self, optimizer: optax.GradientTransformationExtraArgs, n_agents: int, env: Environment, **kwargs):
        self.replay_buffer_size = kwargs.get("replay_buffer_size", 10_000)
        self.n_agents = n_agents
        self.env = env
        self.observation_size = self.env.get_observation_size()

        # Initialize Random Number Generators for NNX
        self.rngs = nnx.Rngs(0)

        # Primary Q-Network (The one we update via gradients)
        self.value_networks = [QNetwork(self.observation_size, 5, rngs=nnx.Rngs(i)) for i in range(n_agents)]

        # Target Q-Network (The stable reference for calculating TD targets)
        self.target_networks = [QNetwork(self.observation_size, 5, rngs=nnx.Rngs(i + n_agents)) for i in range(n_agents)]

        # Synchronize target network weights with the primary network immediately
        self._sync_target_network()

        # Initialize separate replay buffers for each independent agent
        self.replay_buffers = [
            ReplayBuffer(
                self.replay_buffer_size,
                env_dict={"obs": {"shape": self.observation_size}, "act": {}, "rew": {}, "next_obs": {"shape": self.observation_size}, "done": {}},
            )
            for _ in range(n_agents)
        ]

        # NNX Optimizer handles parameter updates for the value network
        # self.optimizer = nnx.Optimizer(self.value_network, optimizer, wrt=nnx.Param)
        self.optimizers = [nnx.Optimizer(value_network, optimizer, wrt=nnx.Param) for value_network in self.value_networks]

    def _sync_target_network(self):
        """Copies weights from value_network to target_network to stabilize learning."""
        for target_network, value_network in zip(self.target_networks, self.value_networks):
            _, state = nnx.split(value_network, nnx.Param)
            nnx.update(target_network, state)

    def save(self, name: str = "latest_model"):
        """Saves the value network using your project's utilities."""
        for i, value_network in enumerate(self.value_networks):
            save_checkpoint(value_network, f"{name}_{i}")
            logger.info(f"Checkpoint '{name}_{i}' saved to disk.")

    def epsilon_update(self, epsilon: float, epsilon_min: float, epsilon_decay: float) -> float:
        return max(epsilon_min, epsilon * epsilon_decay)

    def epsilon_greedy_actions(self, observations, epsilon):
        actions = jnp.zeros(self.n_agents, dtype=jnp.int32)

        for agent in range(self.n_agents):
            if self.rngs.uniform() < epsilon:
                action = self.rngs.randint((), minval=0, maxval=5)
            else:
                q_values = self.value_networks[agent](observations[agent])
                action = jnp.argmax(q_values)
            actions = actions.at[agent].set(action)

        return actions

    def train_step(self, agent, batch_size, discount):
        mini_batch = self.replay_buffers[agent].sample(batch_size)

        # Use the TARGET network to calculate the 'Next Q' values (Stable Target)
        # Temporal Difference (TD) Target: R + gamma * max(Q_target(s', a'))
        # For terminal states, the target is just the reward.
        # Also Q(s, argmax(Q)) for reducing maximisation bias.
        max_next_q = jnp.take_along_axis(
            self.target_networks[agent](mini_batch["next_obs"]),
            jnp.argmax(self.value_networks[agent](mini_batch["next_obs"]), axis=1, keepdims=True),
            axis=1,
        )

        # Don't update parameters for inference of target
        y_target = jax.lax.stop_gradient(mini_batch["rew"].squeeze() + (1.0 - mini_batch["done"].squeeze()) * discount * max_next_q)

        def loss_fn(model, obs, actions, targets):
            q_values = model(obs)
            act_indices = actions.astype(jnp.int32)
            # Extract the Q-values for the actions actually taken
            q_selected = jnp.take_along_axis(q_values, act_indices, axis=1).squeeze()
            return jnp.mean((q_selected - targets) ** 2)

        # Calculate gradients and update the Primary (Value) Network
        loss, grads = nnx.value_and_grad(loss_fn)(self.value_networks[agent], mini_batch["obs"], mini_batch["act"], y_target)
        self.optimizers[agent].update(self.value_networks[agent], grads)


    def train(self, **kwargs):
        n_episodes = kwargs.get("n_episodes", 50)
        epsilon = kwargs.get("epsilon", 0.5)
        epsilon_min = kwargs.get("epsilon_min", 0.05)
        epsilon_decay = kwargs.get("epsilon_decay", 0.99)  # Slowed decay for better exploration
        discount = kwargs.get("discount", 0.99)
        batch_size = kwargs.get("batch_size", 64)
        target_update_freq = kwargs.get("target_update_freq", 100)  # Sync every 100 steps

        total_steps = 0

        for episode in range(n_episodes):
            self.env.reset()
            done = False
            episode_reward = 0.0

            epsilon = self.epsilon_update(epsilon, epsilon_min, epsilon_decay)

            while not done:
                observations = self.env.get_observations()
                actions = self.epsilon_greedy_actions(observations, epsilon)

                _, reward, terminated, truncated = self.env.step(actions)
                episode_reward += float(reward)

                done = jnp.logical_or(jnp.any(terminated), jnp.any(truncated))
                next_observations = self.env.get_observations()

                # Experience replay and learning for each agent independently
                for agent in range(self.n_agents):
                    self.replay_buffers[agent].add(
                        obs=observations[agent],
                        act=actions[agent],
                        rew=reward,
                        next_obs=next_observations[agent],
                        done=done,
                    )

                    if self.replay_buffers[agent].get_stored_size() >= batch_size:
                        self.train_step(agent, batch_size, discount)

                    total_steps += 1
                    if total_steps % target_update_freq == 0:
                        self._sync_target_network()

            # --- Episode Logging ---
            logger.info(f"Episode {episode + 1:3d}/{n_episodes} | Reward: {episode_reward:8.2f} | Epsilon: {epsilon:.3f}")
