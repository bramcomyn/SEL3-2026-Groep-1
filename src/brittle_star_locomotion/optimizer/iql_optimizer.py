import time

from brittle_star_locomotion.config.configuration import Configuration
from brittle_star_locomotion.environment.environment import Environment
from brittle_star_locomotion.neural.qnetwork import QNetwork

from cpprb import ReplayBuffer
from flax import nnx
import optax
import jax
import jax.numpy as jnp


class IQLOptimizer:
    def __init__(self, environment: Environment):
        self._config = Configuration().configuration
        self._seed = self._config.rl.seed # TODO: config
        self._rng = jax.random.PRNGKey(self._seed)

        self._epsilon = self._config.rl.epsilon # TODO: config
        self._environment = environment
        self._observation_size = 16 # TODO: actual observation size

        self._replay_buffers = [
            self._create_replay_buffer() 
            for _ in range(self._config.rl.num_agents) # TODO: config
        ]
        self._q_networks = [
            self._create_qnetwork(agent_id) 
            for agent_id in range(self._config.rl.num_agents) # TODO: config
        ]
        self._target_q_networks = [
            self._create_qnetwork(agent_id)
            for agent_id in range(self._config.rl.num_agents) # TODO: config
        ]
        self._synchronize_target_networks()

        optimizer = optax.chain(
            optax.adam(self._config.rl.learning_rate) # TODO: config
        )
        self._q_network_optimizers = [
            nnx.Optimizer(q_network, optimizer, wrt=nnx.Param)
            for q_network in self._q_networks
        ]

    def optimize(self):
        n_environments = self._config.env.n_environments # TODO: config
        total_train_steps = 0

        for _ in range(self._config.rl.n_episodes): # TODO: config
            self._environment.reset()
            self._update_epsilon()
            environment_is_done = False
            reward = jnp.zeros((n_environments,)) # Reward per environment -- shape (n_envs,)

            while not environment_is_done:
                observations = self._environment.get_observations() # shape (n_envs, n_agents, observation_size)
                actions = self._epsilon_greedy(observations)        # shape (n_envs, n_agents)

                _, reward, terminated, truncated, _ = self._environment.step(actions)

                environment_is_done = bool(jnp.logical_or(jnp.any(terminated), jnp.any(truncated)))
                episode_reward += reward                                 # shape (n_envs,)
                next_observations = self._environment.get_observations() # shape (n_envs, n_agents, observation_size)

                for agent_id in range(self._config.rl.num_agents): # TODO: config
                    for environment_id in range(self._config.rl.n_envs): # TODO: config
                        self._replay_buffers[agent_id].add(
                            obs=observations[environment_id, agent_id],
                            act=actions[environment_id, agent_id],
                            rew=reward[environment_id],
                            next_obs=next_observations[environment_id, agent_id],
                            done=bool(terminated[environment_id] | truncated[environment_id])
                        )

                    if self._replay_buffers[agent_id].get_stored_size() >= self._config.rl.batch_size: # TODO: config
                        self._optimize_step(agent_id)

                        total_train_steps += 1
                        if total_train_steps % self._config.rl.target_update_freq == 0: # TODO: config
                            self._synchronize_target_networks()

    def save_model(self, base_name: str = 'model'):
        """Saves the model parameters of all Q-networks to disk with a timestamped filename.

        The filename is constructed with:
        - `base_name`: A base name for the model (default is 'model').
        - The number of agents and segments from the configuration.
        - The agent identifier (e.g., 'agent_0', 'agent_1', etc.).
        - A timestamp to ensure uniqueness.
        """
        for agent_id, q_network in enumerate(self._q_networks):
            config = f'{self._config.rl.n_agents}_agents_{self._config.env.n_segments}_segments' # TODO: config
            agent = f'agent_{agent_id}'
            timestamp = time.strftime("%Y%m%d-%H%M%S")

            model_name = f'{base_name}_{config}_{agent}_{timestamp}'
            q_network.save_checkpoint(model_name)

    def _optimize_step(self, agent_id: int):
        """Performs a single optimization step for the specified agent using the provided mini-batch of experiences.

        This method computes the loss and gradients for the Q-network of the specified agent and updates its parameters accordingly.

        :param agent_id: The identifier of the agent for which to perform the optimization step.
        """
        mini_batch = self._replay_buffers[agent_id].sample(self._config.rl.batch_size) # TODO: config

        # Double Q-learning
        target_q_values = self._target_q_networks[agent_id](mini_batch['next_observation']) # shape (batch_size, n_actions)
        q_values = self._q_networks[agent_id](mini_batch['next_observation'])               # shape (batch_size, n_actions)

        greedy_actions = jnp.argmax(q_values, axis=-1, keepdims=True)   # shape (batch_size, 1)

        max_next_q_values = jnp.take_along_axis(target_q_values, greedy_actions, axis=-1)

        targets = mini_batch['reward'] + self._config.rl.gamma * max_next_q_values * (1.0 - mini_batch['done']) # shape (batch_size, 1) # TODO: config
        targets = jax.lax.stop_gradient(targets) # Do not update parameters for inference of target             # shape (batch_size, 1)
        
        loss, grads = nnx.value_and_grad(self._loss)(
            self._q_networks[agent_id], 
            mini_batch['observation'], 
            mini_batch['action'], 
            targets
        )
        self._q_network_optimizers[agent_id].update(self._q_networks[agent_id], grads)

    def _loss(self, q_network, observations, taken_actions, targets):
        """TODO Q update
        """
        q_values = q_network(observations) # shape (batch_size, n_actions)
        action_q_values = jnp.take_along_axis(q_values, taken_actions, axis=-1) # shape (batch_size, 1)
        loss = jnp.mean((action_q_values - targets) ** 2)
        return loss

    def _epsilon_greedy(
        self, 
        observations: jnp.ndarray
    ) -> jnp.ndarray:
        """TODO
        
        :param observations: Current environment observations (n_environments, n_agents, observation_size).
        :return: The selected actions (n_environments, n_agents).
        """
        self._rng, *subkeys = jax.random.split(self._rng, 3)
        probabilities_key, actions_key = subkeys

        random_probabilities = jax.random.uniform(
            key=probabilities_key,
            shape=(self._config.rl.n_envs, self._config.rl.n_agents) # TODO: config
        ) # shape (n_environments, n_agents)

        random_actions = jax.random.randint(
            key=actions_key,
            shape=(self._config.rl.n_envs, self._config.rl.n_agents), # TODO: config
            minval=0,
            maxval=self._config.rl.n_actions # TODO: add "num_agents" to config
        ) # shape (n_environments, n_agents)

        q_values = jax.vmap(
            lambda model, obs: model(obs),
            in_axes=(0, 1), # Iterate over first axis of q_networks and second axis of observations (n_agents)
            out_axes=1 # Stack results into axis 1 (n_agents)
        )(self._q_networks, observations)
        greedy_actions = jnp.argmax(q_values, axis=-1) # shape (n_environments, n_agents)

        epsilon_greedy = jnp.where(
            random_probabilities < self._epsilon,
            random_actions,
            greedy_actions
        ) # shape (n_environments, n_agents)
        return epsilon_greedy

    def _update_epsilon(self):
        """Updates the epsilon value for epsilon-greedy action selection.

        The epsilon value is decayed by multiplying it with `epsilon_decay`, but it will not go below `epsilon_min`.
        """
        self._epsilon = max(
            self._epsilon * self._config.rl.epsilon_decay, # TODO: config
            self._config.rl.epsilon_min # TODO: config
        ) 

    def _synchronize_target_networks(self):
        """Synchronizes the parameters of the target Q-networks with the current Q-networks."""
        for q_net, target_q_net in zip(self._q_networks, self._target_q_networks):
            target_q_net.update_model_parameters(copy_from=q_net)

    def _create_qnetwork(self, agent_id: int = 0) -> QNetwork:
        """Creates a new QNetwork instance with random parameters.

        `agent_id` can be used to ensure different random initializations for different agents, if needed.
        By default it is set to 0, which means all agents will have the same initial parameters unless specified otherwise.

        :param agent_id: An integer identifier for the agent.
        :return: A new QNetwork instance with randomly initialized parameters.
        """
        return QNetwork(
            input_size=self._observation_size,
            output_size=self._config.rl.action_size, # TODO: config
            rngs=nnx.Rngs(agent_id),
            hidden_size=self._config.rl.hidden_size, # TODO: config
            amount_of_hidden_layers=self._config.rl.amount_of_hidden_layers # TODO: config
        )

    def _create_replay_buffer(self) -> ReplayBuffer:
        """Creates a new replay buffer instance.

        The replay buffer is used to store and sample past experiences for training the Q-networks.

        :return: A new ReplayBuffer instance.
        """
        return ReplayBuffer(
            self._config.rl.replay_buffer_size, # TODO: config
            env_dict={
                "observation": { "shape": (self._observation_size) },
                "action": {},
                "reward": {},
                "next_observation": { "shape": (self._observation_size) },
                "done": {}
            }
        )
