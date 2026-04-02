import logging
import time

import jax
import jax.numpy as jnp
import optax
from cpprb import ReplayBuffer
from flax import nnx

import wandb
from brittle_star_locomotion.environment import Environment
from brittle_star_locomotion.neural.checkpoint import save_checkpoint
from brittle_star_locomotion.neural.qnetwork import QNetwork

logger = logging.getLogger(__name__)

class IndependentQLearning:
    def __init__(self, optimizer: optax.GradientTransformationExtraArgs, n_agents: int, env: Environment, **kwargs):
        self.replay_buffer_size = env.config.rl.replay_buffer_size
        self.n_agents = n_agents
        self.env = env
        self.observation_size = self.env.get_observation_size()

        # Initialize Random Number Generators for NNX
        # self.rngs = nnx.Rngs(0)
        self.key = jax.random.PRNGKey(0)

        # TODO: DRY
        # Primary Q-Network (The one we update via gradients)
        self.value_networks = [
            QNetwork(
                input_size=self.observation_size, 
                output_size=self.env.config.rl.action_space_dim, 
                rngs=nnx.Rngs(i),
                hidden_size=self.env.config.rl.hidden_layer_size,
                amount_of_hidden_layers=self.env.config.rl.amount_of_hidden_layers
            ) 
            for i in range(n_agents)
        ]

        # Target Q-Network (The stable reference for calculating TD targets)
        self.target_networks = [
            QNetwork(
                input_size=self.observation_size, 
                output_size=self.env.config.rl.action_space_dim, 
                rngs=nnx.Rngs(i + n_agents),
                hidden_size=self.env.config.rl.hidden_layer_size,
                amount_of_hidden_layers=self.env.config.rl.amount_of_hidden_layers
            ) 
            for i in range(n_agents)
        ]

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
        # actions = jnp.zeros((config.rl.amount_environments, self.n_agents), dtype=jnp.int32)
        actions = []

        self.key, subkey = jax.random.split(self.key) # TODO

        for agent in range(self.n_agents):

            subkey, sk1, sk2 = jax.random.split(subkey, 3) # TODO

            random_values = jax.random.uniform(sk1, (self.env.config.rl.amount_environments))

            actions_for_agent = jnp.where(
                random_values < epsilon,
                jax.random.randint(sk2, shape=(self.env.config.rl.amount_environments,), minval=0, maxval=self.env.config.rl.action_space_dim),
                jnp.argmax(self.value_networks[agent](observations[:, agent]), axis=1)
            )

            actions.append(actions_for_agent)

        return jnp.stack(actions, axis=1)

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
        return loss

    def train(self, **kwargs):
        wandb.init(
            entity="comyn-bram-universiteit-gent",
            project="brittle-star-locomotion",
            config=kwargs,
            name=f"IQL-{self.n_agents}-agents-{time.strftime("%Y%m%d-%H%M%S")}"
        )
        
        num_episodes       = self.env.config.rl.num_episodes
        epsilon            = self.env.config.rl.epsilon
        epsilon_min        = self.env.config.rl.epsilon_min
        epsilon_decay      = self.env.config.rl.epsilon_decay # Slowed decay for better exploration
        discount           = self.env.config.rl.discount
        batch_size         = self.env.config.rl.batch_size
        target_update_freq = self.env.config.rl.target_update_freq # Sync every 100 steps
        
        total_steps = 0

        for episode in range(num_episodes):
            self.env.reset()
            done = False
            episode_reward = jnp.zeros((self.env.config.rl.amount_environments,))

            epsilon = self.epsilon_update(epsilon, epsilon_min, epsilon_decay)

            while not done:
                observations = self.env.get_observations() # (envs, obs_size)
                actions = self.epsilon_greedy_actions(observations, epsilon)

                _, reward, terminated, truncated, _ = self.env.step(actions)
                episode_reward += reward

                done = jnp.logical_or(jnp.any(terminated), jnp.any(truncated))
                next_observations = self.env.get_observations()

                # Experience replay and learning for each agent independently
                for agent in range(self.n_agents):
                    for environment_index in range(self.env.config.rl.amount_environments):
                        self.replay_buffers[agent].add(
                            obs=observations[environment_index, agent],
                            act=actions[environment_index, agent],
                            rew=reward[environment_index],
                            next_obs=next_observations[environment_index, agent],
                            done=terminated[environment_index] | truncated[environment_index]
                        )

                    if self.replay_buffers[agent].get_stored_size() >= batch_size:
                        loss = self.train_step(agent, batch_size, discount)
                        wandb.log({ f"agent_{agent}/loss": float(loss) }, step=total_steps)

                    total_steps += 1
                    if total_steps % target_update_freq == 0:
                        self._sync_target_network()

            # --- Episode Logging ---
            logger.info(f"Episode {episode + 1:3d}/{num_episodes} | Reward: {jnp.average(episode_reward):8.2f} | Epsilon: {epsilon:.3f}")
            wandb.log({
                "episode/reward": jnp.average(episode_reward),
                "episode/epsilon": epsilon,
                "episode/episode_number": episode,
                "episode/success": float(jnp.any(terminated))
            }, step=total_steps)
