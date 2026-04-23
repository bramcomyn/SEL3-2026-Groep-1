from brittle_star_locomotion.config.configuration import Configuration
from brittle_star_locomotion.environment.environment import Environment
from brittle_star_locomotion.logger.logger import Logger
from brittle_star_locomotion.neural.qnetwork import QNetwork

from cpprb import ReplayBuffer
from flax import nnx
import optax
import jax
import jax.numpy as jnp


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

        self._q_networks = self._create_n_qnetworks()
        self._target_q_networks = self._create_n_qnetworks()
        self._synchronize_target_networks()

        optimizer = optax.chain(
            optax.adam(self._config.rl.learning_rate)
        )

        self._q_network_optimizers = [
            nnx.Optimizer(q_network, optimizer, wrt=nnx.Param)
            for q_network in self._q_networks
        ]

    def optimize(self):
        """Optimizes the Q-networks using the IQL algorithm over multiple episodes of interaction with the environment.
        """
        n_environments = self._environment.number_of_environments
        total_train_steps = 0
        environment_steps = 0

        self._logger.initialize_wandb(project="brittle-star-locomotion", config=self._config, enabled=self._config.logging.use_wandb)
        self._logger.info(
            f"Starting IQL training for {self._config.rl.n_episodes} episodes "
            f"across {n_environments} environments"
        )

        for episode_index in range(self._config.rl.n_episodes):
            total_train_steps, environment_steps = self._run_episode(
                episode_index=episode_index + 1,
                n_environments=n_environments,
                total_train_steps=total_train_steps,
                environment_steps=environment_steps,
            )

        self._logger.info(
            f"Finished IQL training after {environment_steps} environment steps and "
            f"{total_train_steps} optimization steps"
        )

    def _run_episode(
        self,
        episode_index: int,
        n_environments: int,
        total_train_steps: int,
        environment_steps: int,
    ) -> tuple[int, int]:
        """Run a single episode across all vectorized environments."""
        env_state, cpg_state = self._environment.reset()
        previous_epsilon = self._epsilon
        self._update_epsilon()
        self._logger.info(
            f"Episode {episode_index}/{self._config.rl.n_episodes} started: "
            f"epsilon {previous_epsilon:.4f} -> {self._epsilon:.4f}"
        )

        done_environments = jnp.zeros((n_environments,), dtype=bool)
        episode_steps = 0
        episode_reward_sum = 0.0
        episode_loss_sum = 0.0
        episode_terminated_count = 0
        episode_truncated_count = 0
        episode_done_count = 0

        while not bool(jnp.all(done_environments)):
            transition = self._collect_transition(
                env_state=env_state,
                cpg_state=cpg_state,
                done_environments=done_environments
            )

            env_state = transition["next_env_state"]
            cpg_state = transition["next_cpg_state"]
            done_environments = transition["done_environments"]

            loss_per_environment = self._compute_step_loss_per_environment(
                observations=transition["observations"],
                actions=transition["actions"],
                rewards=transition["reward"],
                next_observations=transition["next_observations"],
                dones=transition["current_done"],
            )

            self._store_replay_transitions(
                observations=transition["observations"],
                actions=transition["actions"],
                rewards=transition["reward"],
                next_observations=transition["next_observations"],
                current_done=transition["current_done"],
                previously_done=transition["previously_done"],
            )

            optimize_losses, total_train_steps = self._optimize_agents(total_train_steps=total_train_steps)

            step_reward_mean = float(jnp.mean(transition["reward"]))
            step_loss_mean = float(jnp.mean(loss_per_environment))
            episode_reward_sum += step_reward_mean
            episode_loss_sum += step_loss_mean
            episode_terminated_count += int(jnp.sum(transition["terminated"]))
            episode_truncated_count += int(jnp.sum(transition["truncated"]))
            episode_done_count += int(jnp.sum(transition["current_done"]))
            episode_steps += 1

            self._log_step_metrics(
                step=environment_steps,
                rewards=transition["reward"],
                losses=loss_per_environment,
                terminated=transition["terminated"],
                truncated=transition["truncated"],
                done=transition["current_done"],
                optimize_losses=jnp.asarray(optimize_losses) if len(optimize_losses) > 0 else None,
            )

            environment_steps += 1

        self._logger.info(
            f"Episode {episode_index}/{self._config.rl.n_episodes} finished after {episode_steps} steps: "
            f"episode_return={episode_reward_sum:.4f}, "
            f"mean_step_reward={episode_reward_sum / max(1, episode_steps):.4f}, "
            f"mean_step_loss={episode_loss_sum / max(1, episode_steps):.4f}, "
            f"terminated={episode_terminated_count}, truncated={episode_truncated_count}, "
            f"done={episode_done_count}, total_train_steps={total_train_steps}"
        )

        return total_train_steps, environment_steps

    def _collect_transition(self, env_state, cpg_state, done_environments: jnp.ndarray) -> dict:
        """Collect one transition step for all environments with action-masking for finished envs."""
        observations = self._environment.get_observations()
        actions = self._epsilon_greedy(observations)
        previously_done = done_environments

        actions = jnp.where(previously_done[:, None], 0, actions)

        next_env_state, next_cpg_state, reward, terminated, truncated, _ = self._environment.step(
            env_state, cpg_state, actions
        )
        
        # Update environment's internal state for get_observations() calls
        self._environment.env_state = next_env_state
        self._environment.cpg_state = next_cpg_state

        # Reshape reward from (n_envs, 1) to (n_envs,)
        reward = jnp.asarray(reward).reshape(-1)
        terminated = jnp.asarray(terminated).reshape(-1)
        truncated = jnp.asarray(truncated).reshape(-1)

        current_done = jnp.asarray(terminated | truncated).reshape(-1)

        next_done_environments = done_environments | current_done
        next_observations = self._environment.get_observations()

        return {
            "observations": observations,
            "actions": actions,
            "reward": reward,
            "next_observations": next_observations,
            "terminated": terminated,
            "truncated": truncated,
            "current_done": current_done,
            "previously_done": previously_done,
            "done_environments": next_done_environments,
            "next_env_state": next_env_state,
            "next_cpg_state": next_cpg_state,
        }

    def _store_replay_transitions(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        next_observations: jnp.ndarray,
        current_done: jnp.ndarray,
        previously_done: jnp.ndarray,
    ) -> None:
        """Store transitions for active environments in each agent replay buffer."""
        n_environments = self._environment.number_of_environments
        for agent_id in range(self._n_agents):
            for environment_id in range(n_environments):
                if bool(previously_done[environment_id]):
                    continue

                self._replay_buffers[agent_id].add(
                    observation=observations[environment_id, agent_id],
                    action=actions[environment_id, agent_id],
                    reward=rewards[environment_id],
                    next_observation=next_observations[environment_id, agent_id],
                    done=bool(current_done[environment_id])
                )

    def _optimize_agents(self, total_train_steps: int) -> tuple[list[float], int]:
        """Optimize all trainable agents once and return losses with updated train step counter."""
        optimize_losses = []
        for agent_id in range(self._n_agents):
            if self._replay_buffers[agent_id].get_stored_size() >= self._config.rl.batch_size:
                optimize_loss = self._optimize_step(agent_id)
                optimize_losses.append(optimize_loss)

                total_train_steps += 1
                if total_train_steps % self._config.rl.target_update_freq == 0:
                    self._synchronize_target_networks()
                    self._logger.debug(
                        f"Synchronized target networks at train step {total_train_steps}"
                    )

        return optimize_losses, total_train_steps

    def save_model(self, checkpoint_prefix: str = 'model'):
        """Save one checkpoint per agent using <checkpoint_prefix>_<agent_id>.

        :param checkpoint_prefix: Prefix used for all checkpoint filenames.
        """
        for agent_id, q_network in enumerate(self._q_networks):
            model_name = f'{checkpoint_prefix}_{agent_id}'
            q_network.save_checkpoint(model_name)

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

    def _compute_step_loss_per_environment(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        next_observations: jnp.ndarray,
        dones: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute one-step TD loss per environment, averaged over agents."""
        losses_by_agent = []

        for agent_id in range(self._n_agents):
            q_values = self._q_networks[agent_id](observations[:, agent_id, :])
            taken_actions = actions[:, agent_id].astype(int).reshape(-1, 1)
            q_taken = jnp.take_along_axis(q_values, taken_actions, axis=-1).reshape(-1)

            next_q_values_online = self._q_networks[agent_id](next_observations[:, agent_id, :])
            next_actions = jnp.argmax(next_q_values_online, axis=-1, keepdims=True)
            next_q_values_target = self._target_q_networks[agent_id](next_observations[:, agent_id, :])
            next_q = jnp.take_along_axis(next_q_values_target, next_actions, axis=-1).reshape(-1)

            targets = rewards + self._config.rl.gamma * next_q * (1.0 - dones.astype(jnp.float32))
            td_loss = (q_taken - jax.lax.stop_gradient(targets)) ** 2
            losses_by_agent.append(td_loss)

        return jnp.stack(losses_by_agent, axis=0).mean(axis=0)

    def _log_step_metrics(
        self,
        step: int,
        rewards: jnp.ndarray,
        losses: jnp.ndarray,
        terminated: jnp.ndarray,
        truncated: jnp.ndarray,
        done: jnp.ndarray,
        optimize_losses: jnp.ndarray | None,
    ) -> None:
        """Log per-environment and aggregate training metrics to Weights & Biases."""
        log_data = {
            "train/reward_mean": float(jnp.mean(rewards)),
            "train/loss_mean": float(jnp.mean(losses)),
            "train/terminated_count": int(jnp.sum(terminated)),
            "train/truncated_count": int(jnp.sum(truncated)),
            "train/done_count": int(jnp.sum(done)),
            "train/terminated_ratio": float(jnp.mean(terminated.astype(jnp.float32))),
            "train/done_ratio": float(jnp.mean(done.astype(jnp.float32))),
        }

        if optimize_losses is not None:
            log_data["train/optimize_loss_mean"] = float(jnp.mean(optimize_losses))

        for environment_id in range(self._environment.number_of_environments):
            log_data[f"train/reward/env_{environment_id}"] = float(rewards[environment_id])
            log_data[f"train/loss/env_{environment_id}"] = float(losses[environment_id])
            log_data[f"train/terminated/env_{environment_id}"] = int(terminated[environment_id])
            log_data[f"train/truncated/env_{environment_id}"] = int(truncated[environment_id])
            log_data[f"train/done/env_{environment_id}"] = int(done[environment_id])

        self._logger.log_metrics(log_data, step=step)

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

    def _create_n_qnetworks(self) -> list[QNetwork]:
        """Creates `n` QNetwork instances, either shared or separate based on the configuration.

        :param n: The number of QNetwork instances to create (default is 1).
        :return: A list of QNetwork instances.
        """
        if self._config.rl.shared_params:
            q_network = self._create_qnetwork()
            return [q_network] * self._n_agents
        else:
            return [self._create_qnetwork(agent_id) for agent_id in range(self._n_agents)]

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
