from __future__ import annotations

import brittle_star_locomotion.control.control as control
import jax
import jax.numpy as jnp
from biorobot.brittle_star.environment.directed_locomotion.dual import BrittleStarDirectedLocomotionEnvironment
from biorobot.brittle_star.environment.directed_locomotion.shared import BaseEnvState, BrittleStarDirectedLocomotionEnvironmentConfiguration
from biorobot.brittle_star.mjcf.arena.aquarium import AquariumArenaConfiguration, MJCFAquariumArena
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from biorobot.brittle_star.mjcf.morphology.specification.default import default_brittle_star_morphology_specification


class Environment:
    def __init__(
        self,
        num_arms: int,
        num_segments_per_arm: int,
        arena_configuration: AquariumArenaConfiguration,
        environment_configuration: BrittleStarDirectedLocomotionEnvironmentConfiguration,
        control: control.Control,
        backend: str = "MJX",
        rng=jax.random.PRNGKey(0),
        observations: None | list[str] = None,
    ) -> None:
        """Make Brittle Star Directed locomotion environment

        :param num_arms: Amount of arms the brittle star has.
        :type int: num_arms

        :param num_segments_per_arm: Amount of segments per arm the brittle star has.
        :type int: num_segments_per_arm

        :param arena_configuration: Configuration for the arena.
        :type ArenaConfiguration: arena_configuration

        :param environment_configuration: Configuration for the environment.
        :type EnvironmentConfiguration: environment_configuration

        :param control: Controller for stepping in the environment.
        :type Control: control

        :param backend: Backend to use for the environment.
        :type str: backend

        :param rng: Random number generator key.
        :type jax.random.PRNGKey: rng

        :param observations: List of observations to include into the observation space.
        :type list[str]: observations
        """
        self.rng = rng
        self.num_arms = num_arms
        self.num_segments_per_arm = num_segments_per_arm

        self.morphology_specification = default_brittle_star_morphology_specification(
            num_arms=self.num_arms, num_segments_per_arm=self.num_segments_per_arm, use_p_control=True, use_torque_control=False
        )
        self.arena_configuration = arena_configuration
        self.environment_configuration = environment_configuration
        self.morphology = MJCFBrittleStarMorphology(self.morphology_specification)
        self.arena = MJCFAquariumArena(self.arena_configuration)
        self.env = BrittleStarDirectedLocomotionEnvironment.from_morphology_and_arena(
            morphology=self.morphology, arena=self.arena, configuration=self.environment_configuration, backend=backend
        )

        self.control = control
        self.control.init(env=self)

        self.state = self.env.reset(self.rng)
        self.state_space = {
            "actuator_force": 2,
            "disk_angular_velocity": 1,
            "disk_linear_velocity": 1,
            "disk_position": 1,
            "disk_rotation": 1,
            "joint_actuator_force": 2,
            "joint_position": 2,
            "joint_velocity": 2,
            "segment_contact": 1,
            "tendon_position": 0,
            "tendon_velocity": 0,
            "unit_xy_direction_to_target": 2,
            "xy_distance_to_target": 1,
        }

        if observations is None:
            self.observations = list(self.state_space.keys())
        else:
            for obs in observations:
                assert obs in self.state_space, f"Observation {obs} not in state space {self.state_space}"
            self.observations = observations

        self.jit_env_step = jax.jit(self.env.step)
        self.jit_env_reset = jax.jit(self.env.reset)

    def step(self, actions: jnp.ndarray):  # TODO return type
        """Step in the environment.

        :param actions: Actions to take in the environment, one float for each actuator.
        :type jnp.ndarray: actions

        :return: The new state, reward, termination status, truncation status, and info.
        :rtype: tuple
        """
        # self.state = self.jit_env_step(self.state, actions)

        # TODO: typing is a bit all over the place
        self.state: BaseEnvState = self.control(self, actions)
        return self.state, self.state.reward, self.state.terminated, self.state.truncated

    def reset(self):  # TODO return type
        """Reset the environment

        :return: The new state.
        :rtype: BaseEnvState
        """
        self.state = self.jit_env_reset(self.rng)
        return self.state

    def get_observations(self) -> jnp.ndarray:
        """Get the observations from the environment per agent.

        For each agent, this returns the selected observations from `observations` in the constructor.
        Individual observations are concatenated into a single array.
        Observations are then stacked along a new dimension.

        :return: The selected observations per agent with shape (num_agents, observation_space_shape).
        :rtype: jnp.ndarray
        """
        observation_space_shape = self.get_observation_size()
        observations = jnp.zeros((self.num_arms, observation_space_shape))

        for arm in range(self.num_arms):
            i = 0
            for obs in self.observations:
                for obs_idx in range(self.state_space[obs]):
                    observations.at[arm, i].set(self.state.observations[obs][arm * self.state_space[obs] + obs_idx])  # type: ignore
                    i += 1

        return observations

    def get_observation_size(self) -> int:
        """Get the size of the observation space.

        :return: The size of the observation space.
        :rtype: int
        """
        return self.num_segments_per_arm * sum(self.state_space[obs] for obs in self.observations)
