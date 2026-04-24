from brittle_star_locomotion.config.configuration import Configuration
from brittle_star_locomotion.environment.environment import Environment
from brittle_star_locomotion.logger.logger import Logger
from brittle_star_locomotion.neural.qnetwork import QNetwork
from brittle_star_locomotion.optimizer.transition import Transition
from brittle_star_locomotion.metrics.training_metrics import TrainingMetrics

from cpprb import ReplayBuffer
from flax import nnx
import optax
import jax
import jax.numpy as jnp

from typing import Any # TODO remove


class IQLOptimizer:
    def __init__(self, environment: Environment):
        """Initializes an IQL optimizer for multi-agent reinforcement learning.

        Initializes the replay buffers, Q-networks, target Q-networks, and optimizers for each agent 
        based on the provided environment and configuration settings.

        Extensions on basic Q-Learning include:
        - **Double Q-Learning** to reduce maximization bias

        :param environment: An instance of the Environment class that the optimizer will interact with.
        """
        self._config = Configuration().configuration
        self._seed = self._config.rl.seed
        self._rng = jax.random.PRNGKey(self._seed)

        self._epsilon = self._config.rl.epsilon
        self._environment = environment
        self._observation_size = self._environment.get_observation_size()
        self._n_agents = self._environment.number_of_arms
        self._n_environments = self._environment.number_of_environments
        self._n_actions = 5
        self._done_environments = jnp.zeros((self._n_environments,), dtype=bool) # shape (n_envs,)

        self._metrics = TrainingMetrics()

        self._logger = Logger()
        self._logger.info(
            "IQL optimizer initialized with "
            f"{self._n_agents} agents, {self._n_environments} environments, "
            f"{self._n_actions} actions, observation size {self._observation_size}, "
            f"episodes={self._config.rl.n_episodes}, batch_size={self._config.rl.batch_size}, "
            f"target_update_freq={self._config.rl.target_update_freq}"
        )
        self._logger.info(
            f"Training hyperparameters: seed={self._seed}, epsilon={self._epsilon:.4f}, "
            f"epsilon_decay={self._config.rl.epsilon_decay}, epsilon_min={self._config.rl.epsilon_min}, "
            f"learning_rate={self._config.rl.learning_rate}, gamma={self._config.rl.gamma}"
        )

        self._replay_buffers = self._create_replay_buffers()

        self._q_networks = self._create_qnetworks()
        self._target_q_networks = self._create_qnetworks()
        self._synchronize_target_networks()

        optimizer = optax.chain(
            optax.adam(self._config.rl.learning_rate)
        )

        self._q_network_optimizers = [
            nnx.Optimizer(q_network, optimizer, wrt=nnx.Param)
            for q_network in self._q_networks
        ]

    def save_model(self, checkpoint_prefix: str = 'model') -> None:
        """Save one checkpoint per agent using <checkpoint_prefix>_<agent_id>.

        :param checkpoint_prefix: Prefix used for all checkpoint filenames.
        """
        for agent_id, q_network in enumerate(self._q_networks):
            model_name = f'{checkpoint_prefix}_{agent_id}'
            q_network.save_checkpoint(model_name)

    def optimize(self) -> None:
        """Optimizes the Q-networks using the IQL algorithm over multiple episodes of interaction with the environment."""
        self._metrics.new_training()

        for _ in range(self._config.rl.n_episodes):
            self._done_environments = jnp.zeros((self._n_environments,), dtype=bool) # shape (n_envs,)
            self._update_epsilon()
            self._run_episode()

        self._metrics.end_training()

    def _run_episode(self) -> None:
        """Run a single episode across all vectorized environments."""
        env_state, cpg_state = self._environment.reset()

        self._metrics.new_episode(self._epsilon)

        while not bool(jnp.all(self._done_environments)):
            transition, env_state, cpg_state = self._step_in_environment(env_state, cpg_state)
            self._store_replay_transitions(transition)    
            losses = self._optimize_agents()
            self._metrics.new_episode_step(transition, losses)

        self._metrics.end_episode()

    def _step_in_environment(self, env_state, cpg_state) -> tuple[Transition, Any, Any]:  # TODO remove Any
        """Collect one transition step for all environments.

        Makes observations of the state for each agent and determines the epsilon-greedy actions.
        It steps once in the environment with these actions.

        :return: The transition tuple from stepping once through the environment
        """
        observations = self._environment.get_observations() # shape (n_environments, n_agents, obs)
        actions = self._epsilon_greedy(observations)        # shape (n_environments, n_agents)

        # Puts all actions to 0 for environments that are done
        actions = jnp.where(self._done_environments[:, None], 0, actions) # shape (n_environments, n_agents)

        next_env_state, next_cpg_state, reward, terminated, truncated, _ = self._environment.step(
            env_state, cpg_state, actions
        )

        # Update environment's internal state for get_observations() calls
        self._environment.env_state = next_env_state
        self._environment.cpg_state = next_cpg_state

        self._done_environments = self._done_environments | terminated | truncated
        next_observations = self._environment.get_observations()

        transition = Transition(
            observations,
            actions,
            reward,
            next_observations,
            terminated,
            truncated
        )
        return transition, next_env_state, next_cpg_state

    def _store_replay_transitions(
        self,
        transition: Transition
    ) -> None:
        """Store transitions for non-terminated or truncated environments in each agent replay buffer.

        :param transition: The transition to store
        """
        for agent_id in range(self._n_agents):
            for environment_id in range(self._n_environments):
                if not bool(self._done_environments[environment_id]):
                    done = bool(transition.terminated[environment_id] | transition.truncated[environment_id])
                    self._replay_buffers[agent_id].add(
                        observation=transition.observations[environment_id, agent_id],
                        action=transition.actions[environment_id, agent_id],
                        reward=transition.rewards[environment_id],
                        next_observation=transition.next_observations[environment_id, agent_id],
                        done=done
                    )

    def _optimize_agents(self) -> jnp.ndarray:
        """Optimize all agents once by sampling a minibatch from their replay buffers and return losses.
        
        :return: Losses from the QNetworks.
        """
        losses = []
        for agent_id in range(self._n_agents):
            if self._replay_buffers[agent_id].get_stored_size() >= self._config.rl.batch_size:
                loss = self._optimize_step(agent_id)
                losses.append(loss)

                self._total_train_steps += 1
                if self._total_train_steps % self._config.rl.target_update_freq == 0:
                    self._synchronize_target_networks()
                    self._logger.debug(
                        f"Synchronized target networks at train step {self._total_train_steps}"
                    )
            else:
                losses.append(0)

        return jnp.array(losses, dtype=jnp.float32)

    def _optimize_step(self, agent_id: int):
        """Performs a single optimization step for the specified agent.

        Optimizes the Q-network for the given `agent_id` by sampling a mini-batch of experiences from its replay buffer,
        and doing the standard Q-learning update using the Double Q-learning approach to compute the target Q-values.
        This method computes the loss and gradients for the Q-network of the specified agent and updates its parameters accordingly.

        :param agent_id: The identifier of the agent for which to perform the optimization step.
        """
        mini_batch = self._replay_buffers[agent_id].sample(self._config.rl.batch_size)

        # Double Q-learning
        target_q_values = self._target_q_networks[agent_id](mini_batch['next_observation']) # shape (batch_size, n_actions)
        q_values = self._q_networks[agent_id](mini_batch['next_observation'])               # shape (batch_size, n_actions)

        greedy_actions = jnp.argmax(q_values, axis=-1, keepdims=True)   # shape (batch_size, 1)

        max_next_q_values = jnp.take_along_axis(target_q_values, greedy_actions, axis=-1)

        targets = mini_batch['reward'] + self._config.rl.gamma * max_next_q_values * (1.0 - mini_batch['done']) # shape (batch_size, 1)
        targets = jax.lax.stop_gradient(targets) # Do not update parameters for inference of target             # shape (batch_size, 1)
        
        loss, grads = nnx.value_and_grad(self._loss)(
            self._q_networks[agent_id], 
            mini_batch['observation'], 
            mini_batch['action'], 
            targets
        )

        self._q_network_optimizers[agent_id].update(self._q_networks[agent_id], grads)
        return float(loss)

    def _loss(
        self, 
        q_network: QNetwork,
        observations: jnp.ndarray,
        taken_actions: jnp.ndarray,
        targets: jnp.ndarray
    ) -> jnp.ndarray:
        """Computes the loss for the Q-network.

        :param q_network: The Q-network for which to compute the loss.
        :param observations: The observations for which to compute the Q-values.
        :param taken_actions: The actions that were taken.
        :param targets: The target Q-values.
        :return: The computed loss.
        """
        q_values = q_network(observations) # shape (batch_size, n_actions)
        # Cast actions to int if needed (replay buffer stores as float)
        taken_actions = taken_actions.astype(jnp.int32)
        action_q_values = jnp.take_along_axis(q_values, taken_actions, axis=-1) # shape (batch_size, 1)
        loss = jnp.mean((action_q_values - targets) ** 2)
        return loss.astype(float)

    def _epsilon_greedy(
        self, 
        observations: jnp.ndarray
    ) -> jnp.ndarray:
        """Determines the epsilon-greedy action for each agent for the given observations of the environment.

        :param observations: Current environment observations (n_environments, n_agents, observation_size).
        :return: The selected actions (n_environments, n_agents).
        """
        self._rng, *subkeys = jax.random.split(self._rng, 3)
        probabilities_key, actions_key = subkeys

        random_probabilities = jax.random.uniform(
            key=probabilities_key,
            shape=(self._n_environments, self._n_agents)
        ) # shape (n_environments, n_agents)

        random_actions = jax.random.randint(
            key=actions_key,
            shape=(self._n_environments, self._n_agents),
            minval=0,
            maxval=self._n_actions
        ) # shape (n_environments, n_agents)

        q_values_list = []
        for agent_id in range(self._n_agents):
            agent_obs = observations[:, agent_id, :]
            agent_q_values = self._q_networks[agent_id](agent_obs)  # shape (n_envs, n_actions)
            q_values_list.append(agent_q_values)

        q_values = jnp.stack(q_values_list, axis=1)  # shape (n_envs, n_agents, n_actions)
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
            self._epsilon * self._config.rl.epsilon_decay,
            self._config.rl.epsilon_min
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
            output_size=self._n_actions,
            rngs=nnx.Rngs(agent_id),
            hidden_size=self._config.rl.hidden_size,
            amount_of_hidden_layers=self._config.rl.amount_of_hidden_layers
        )

    def _create_qnetworks(self) -> list[QNetwork]:
        """Creates `n` QNetwork instances, either shared or separate based on the configuration.

        :param n: The number of QNetwork instances to create (default is 1).
        :return: A list of QNetwork instances.
        """
        if self._config.rl.shared_params:
            q_network = self._create_qnetwork()
            return [q_network] * self._n_agents
        else:
            return [self._create_qnetwork(agent_id) for agent_id in range(self._n_agents)]

    def _create_replay_buffer(self) -> ReplayBuffer:
        """Creates a new replay buffer instance.

        The replay buffer is used to store and sample past experiences for training the Q-networks.

        :return: A new ReplayBuffer instance.
        """
        return ReplayBuffer(
            self._config.rl.replay_buffer_size,
            env_dict={
                "observation": { "shape": (self._observation_size) },
                "action": {},
                "reward": {},
                "next_observation": { "shape": (self._observation_size) },
                "done": {}
            }
        )
    
    def _create_replay_buffers(self) -> list[ReplayBuffer]:
        """Creates a list of replay buffers, one for each agent.
        If `shared_params` is True in the configuration, all agents will share the same replay buffer instance.

        :return: A list of ReplayBuffer instances, one for each agent.
        """
        if self._config.rl.shared_params:
            shared_buffer = self._create_replay_buffer()
            return [shared_buffer] * self._n_agents
        else:
            return [self._create_replay_buffer() for _ in range(self._n_agents)]
