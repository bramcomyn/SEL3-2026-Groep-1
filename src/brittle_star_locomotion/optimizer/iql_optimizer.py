import time

from brittle_star_locomotion.environment.environment import Environment
from brittle_star_locomotion.neural.qnetwork import QNetwork

from cpprb import ReplayBuffer
from flax import nnx
import optax
import jax


class IQLOptimizer:
    def __init__(self, environment: Environment):
        self._environment = environment

        self._config = None # TODO: load config file

        self._observation_size = 16 # TODO: actual observation size

        self._replay_buffers = [
            self._create_replay_buffer() 
            for _ in range(self._config.rl.num_agents)
        ]

        self._seed = 42 # TODO: load seed from config file
        self._rng = jax.random.PRNGKey(self._seed)

        self._q_networks = [
            self._create_qnetwork(agent_id) 
            for agent_id in range(self._config.rl.num_agents)
        ]

        self._target_q_networks = [
            self._create_qnetwork(agent_id)
            for agent_id in range(self._config.rl.num_agents)
        ]

        self._synchronize_target_networks()

        optimizer = optax.chain(
            optax.adam(self._config.rl.learning_rate)
        )
        self._qnetwork_optimizers = [
            nnx.Optimizer(q_network, optimizer, wrt=nnx.Param)
            for q_network in self._q_networks
        ]

    def optimize(self):
        pass

    def save_model(self, base_name: str = 'model'):
        """Saves the model parameters of all Q-networks to disk with a timestamped filename.

        The filename is constructed with:
        - `base_name`: A base name for the model (default is 'model').
        - The number of agents and segments from the configuration.
        - The agent identifier (e.g., 'agent_0', 'agent_1', etc.).
        - A timestamp to ensure uniqueness.
        """
        for agent_id, q_network in enumerate(self._q_networks):
            config = f'{self._config.rl.n_agents}_agents_{self._config.env.n_segments}_segments'
            agent = f'agent_{agent_id}'
            timestamp = time.strftime("%Y%m%d-%H%M%S")

            model_name = f'{base_name}_{config}_{agent}_{timestamp}'
            q_network.save_checkpoint(model_name)

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
            output_size=self._config.rl.action_size,
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
