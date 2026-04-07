import jax
import jax.numpy as jnp
import functools

from biorobot.brittle_star.mjcf.morphology.specification.default import default_brittle_star_morphology_specification
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from biorobot.brittle_star.mjcf.arena.aquarium import AquariumArenaConfiguration, MJCFAquariumArena
from biorobot.brittle_star.environment.directed_locomotion.shared import BrittleStarDirectedLocomotionEnvironmentConfiguration
from biorobot.brittle_star.environment.directed_locomotion.dual import BrittleStarDirectedLocomotionEnvironment, DualMuJoCoEnvironment

from brittle_star_locomotion.controller.rowing_gait_controller import RowingGaitController
from brittle_star_locomotion.config.configuration import Configuration

# TODO: 
# - step
# - add types to all functions (parameters and return types)
# - add docstrings to all functions


class Environment:
    def __init__(self):
        self.config = Configuration().configuration

        self.number_of_environments = self.config.rl.number_of_environments # TODO: put number_of_environments in config
        self.number_of_arms = self.config.env.num_arms # TODO: add number_of_arms to configuration file
        self._number_of_segments_per_arm = self.config.env.num_segments_per_arm # TODO: add number_of_segments_per_arm to configuration file

        # Construct the morphology, arena, and environment based on the configuration
        self.morphology = self._create_morphology()
        self.arena = self._create_arena()
        self.brittle_star_environment = self._create_brittle_star_directed_locomotion_environment()

        # Construct CPG
        self.controller = RowingGaitController()
        self.cpg_state = self.controller.cpg.state

        self.rng = jax.random.PRNGKey(seed=self.config.env.seed) # TODO: put seed in config
        self.rng, reset_key, *self.sub_rngs = jnp.array(jax.random.split(self.rng, self.number_of_environments + 2))
        
        self.max_joint_limit = self.brittle_star_environment.action_space.high[0] # type: ignore
        # Observations
        self.derived_states = ["arm_identification", "angle_to_target"]
        self.state_space = {
            "central": {
                "disk_angular_velocity": 3,
                "disk_linear_velocity": 3,
                "disk_position": 3,
                "disk_rotation": 3,
                # Specific to the directed locomotion task
                "unit_xy_direction_to_target": 2,
                "xy_distance_to_target": 1,
            },
            "individual_per_segment": {
                "actuator_force": 2,
                "joint_actuator_force": 2,
                "joint_position": 2,
                "joint_velocity": 2,
            },
            "individual_per_arm": {
                "segment_contact": 3,
                "angle_to_target": 1,
                "arm_identification": self.number_of_arms,
            },
        }
        self.observations_to_use = self._get_observations_to_use()

        self.jit_env_step = jax.jit(jax.vmap(self.brittle_star_environment.step))
        self.jit_env_reset = jax.jit(jax.vmap(self.brittle_star_environment.reset))
        self.jit_env_reset_single = jax.jit(self.brittle_star_environment.reset)

        self.env_state = self._reset_all_envs(jnp.array(self.sub_rngs))


    def _reset_all_envs(self, sub_rngs: jnp.ndarray):
        """Reset all environments while avoiding vmapped-reset instability for single-env runs."""
        if self.number_of_environments == 1:
            single_env_state = self.jit_env_reset_single(sub_rngs[0])
            return jax.tree_util.tree_map(lambda x: x[jnp.newaxis, ...], single_env_state)
        return self.jit_env_reset(sub_rngs)

    def reset(self):
        """Reset both the MJX environment and the CPG controllers."""
        self.env_state = self._reset_all_envs(jnp.array(self.sub_rngs))
        self.cpg_state = self.controller.cpg.reset()
        return self.env_state, self.cpg_state
        
    @functools.partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        env_state,
        cpg_state,
        action: jnp.ndarray,
        num_substeps: int = 50,
    ):
        """Modulate the rowing gait with arm-role actions and run CPG+physics substeps.

        :param env_state: current environment state.
        :param cpg_state: current CPG state.
        :param action: arm-role modulation action with shape (arms,) or (envs, arms).
        :param num_substeps: number of CPG/physics substeps to execute per call.
        """
        previous_distance: jnp.ndarray = env_state.observations["xy_distance_to_target"] # shape: (envs, 1) # type: ignore

        if action.ndim == 1:
            action = jnp.broadcast_to(action[jnp.newaxis, :], (self.number_of_environments, action.shape[0]))

        cpg_state = self.controller.modulate(cpg_state, action, self.max_joint_limit)

        def _cpg_loop_body(carry, _):
            _env_state, _cpg_state = carry
            _next_cpg_state, _next_cpg_action = self.controller.step(_cpg_state)
            _next_env_action = self._map_cpg_output_to_environment_action(_next_cpg_action)
            _next_env_state = self.jit_env_step(_env_state, _next_env_action)
            return (_next_env_state, _next_cpg_state), (_next_env_state, _next_cpg_action)

        (next_env_state, next_cpg_state), trajectory = jax.lax.scan(
            _cpg_loop_body,
            (env_state, cpg_state),
            None,
            length=num_substeps,
        )

        current_distance = next_env_state.observations["xy_distance_to_target"] # shape: (envs, 1)

        reward = (previous_distance - current_distance)             # shape: (envs, 1)
        terminated = jnp.asarray(next_env_state.terminated).reshape(-1, 1)
        reward += 10.0 * terminated                                  # shape: (envs, 1)
        reward = reward.reshape(-1)                                 # shape: (envs,)

        return next_env_state, next_cpg_state, reward, next_env_state.terminated, next_env_state.truncated, trajectory

    @functools.partial(jax.jit, static_argnums=(0,))
    def _map_cpg_output_to_environment_action(self, cpg_output: jnp.ndarray) -> jnp.ndarray:
        """Map one IP/OOP pair per arm to per-segment actuator controls expected by biorobot."""
        ip = cpg_output[:, 0::2]
        oop = cpg_output[:, 1::2]
        per_arm = jnp.stack([ip, oop], axis=-1)  # (envs, arms, 2)
        per_segment = jnp.repeat(per_arm[:, :, jnp.newaxis, :], self._number_of_segments_per_arm, axis=2)
        return per_segment.reshape(cpg_output.shape[0], -1)

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_observations(self) -> jnp.ndarray:  # (envs, arms, total_obs_per_arm)
        """Construct the observation tensor for all environments and arms based on the specified observations to use in the configuration.
        This function loops through the observations specified in the configuration, retrieves the corresponding data from the environment state, 
        and constructs a tensor of shape (num_envs, num_arms, total_obs_per_arm) that contains the observations for each arm in each environment. 
        It also handles derived observations like the angle from each arm to the target. 

        :return: observation tensor for all environments and arms with shape (num_envs, num_arms, total_obs_per_arm)
        :rtype: jnp.ndarray
        """

        obs_list = []
        num_envs = self.number_of_environments
        num_arms = self.number_of_arms

        # loop through all observations we want to use and construct the observation tensor
        for obs in self.observations_to_use:

            # if the obsercation is not derived, get it from the environment state 
            # and reshape it to be (envs, arms, obs_per_arm)
            if obs not in self.derived_states:

                if obs in self.state_space["central"]:
                    data = self.env_state.observations[obs][:, jnp.newaxis, :] # type: ignore       # (envs, 1, dim)    - add an extra dimension so the observation can be duplicated across all arms, since this is a central observation that is the same for each arm
                    data = jnp.broadcast_to(data, (num_envs, num_arms, data.shape[-1]))             # (envs, arms, dim) - broadcast the central observation to be the same for each arm
                    obs_list.append(data)

                elif obs in self.state_space["individual_per_segment"]:
                    data = self.env_state.observations[obs]                 # (envs, arms * segments * dim) - these observations are concatenated for all segments in an arm
                    data = data.reshape(num_envs, num_arms, -1)             # (envs, arms, segments * dim)  - reshape the observation to be separate for each arm, but still concatenated for all segments in an arm
                    obs_list.append(data)

                elif obs in self.state_space["individual_per_arm"]:
                    data = self.env_state.observations[obs]                 # (envs, arms * dim) - these observations are concatenated for all arms, but not for segments
                    data = data.reshape(num_envs, num_arms, -1)             # (envs, arms, dim)  - reshape the observation to be separate for each arm
                    obs_list.append(data)

            elif obs == "angle_to_target":
                relative_angle_arm_to_target = self._get_angle_arm_to_target()      # shape: (envs, arms, 1)
                obs_list.append(relative_angle_arm_to_target)

        return jnp.concatenate(obs_list, axis=-1)

    def get_observation_size(self) -> int:
        """Calculate the total size of the observation vector for each arm based on the specified observations to use in the configuration and the state space.
        :return: total size of the observation vector for each arm
        """
        size = 0

        for obs in self.observations_to_use:
            if obs in self.state_space["central"]:
                size += self.state_space["central"][obs]

            elif obs in self.state_space["individual_per_segment"]:
                size_per_segment = self.state_space["individual_per_segment"][obs]
                size += self.config.env.num_segments_per_arm * size_per_segment

            elif obs in self.state_space["individual_per_arm"]:
                size += self.state_space["individual_per_arm"][obs]

        return size

    def _create_morphology(self) -> MJCFBrittleStarMorphology:
        """Construct the brittle star morphology based on the configuration."""
        morphology_specification = default_brittle_star_morphology_specification(
            num_arms=self.number_of_arms, 
            num_segments_per_arm=self.config.env.num_segments_per_arm, 
            use_p_control=True, 
            use_torque_control=False
        )
        return MJCFBrittleStarMorphology(morphology_specification)
    
    def _create_arena(self) -> MJCFAquariumArena:
        """Construct the aquarium arena based on the configuration."""
        arena_configuration = AquariumArenaConfiguration(
            size=(self.config.env.arena.size_x, self.config.env.arena.size_y), 
            sand_ground_color=self.config.env.arena.sand_ground_color, 
            attach_target=self.config.env.arena.attach_target, 
            wall_height=self.config.env.arena.wall_height, 
            wall_thickness=self.config.env.arena.wall_thickness
        )
        return MJCFAquariumArena(arena_configuration)
    
    def _create_brittle_star_directed_locomotion_environment(self) -> DualMuJoCoEnvironment:
        """Construct the brittle star directed locomotion environment based on the morphology, arena, and configuration."""
        environment_configuration = BrittleStarDirectedLocomotionEnvironmentConfiguration(
            target_distance=self.config.env.target_distance,
            joint_randomization_noise_scale=0.0,
            render_mode="rgb_array",
            simulation_time=self.config.env.simulation_time,
            num_physics_steps_per_control_step=self.config.env.num_physics_steps_per_control_step,
            time_scale=self.config.env.time_scale,
            camera_ids=self.config.env.camera_ids,
            render_size=(self.config.env.render_size_x, self.config.env.render_size_y),
        )

        return BrittleStarDirectedLocomotionEnvironment.from_morphology_and_arena(
            self.morphology, self.arena, environment_configuration, "MJX"
        )
    
    def _get_observations_to_use(self) -> list[str]:
        """Determine which observations to use based on the configuration and validate them against the state space.
        If no observations are specified in the configuration, use all available observations from the state space.
        """
        observations_to_use_from_config = self.config.rl.observations_to_use

        valid_keys = (
            list(self.state_space["central"].keys()) +
            list(self.state_space["individual_per_segment"].keys()) +
            list(self.state_space["individual_per_arm"].keys())
        )

        if observations_to_use_from_config is None: 
            # if no observations are specified, use all observations
            return valid_keys
        else:
            # If observations are specified, validate them and use them
            for obs in observations_to_use_from_config:
                assert obs in valid_keys, f"Observation {obs} not in state space."
            return observations_to_use_from_config

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_angle_arm_to_target(self) -> jnp.ndarray:
        """Calculate relative angle from each arm to the target

        This function calculates the relative angle from each arm of the brittle star to the target, taking into account the absolute rotation of the body and the position of the target.
        The steps are as follows:
        1. Get the absolute rotation of the body from the observations.
        2. Calculate the relative rotation of each arm based on the number of arms and their equal spacing around the body.
        3. Get the absolute position of the body and the direction and distance to the target from the observations.
        4. Calculate the absolute position of the target in the world frame.
        5. Calculate the absolute angle from the body to the target using the arctangent of the target's position.
        6. Calculate the relative angle from the body to the target by subtracting the body's absolute rotation from the absolute angle to the target.
        7. Calculate the relative angle from each arm to the target by subtracting the relative arm rotation from the relative angle from the body to the target, and normalize it to the range [-pi, pi].

        When talking about the absolute rotation or position, this is in world frame.
        When talking about the relative rotation of the arms, this is relative to the body of the brittle star (so it does not change as the body rotates).
        When talking about the relative angle from the arm to the target, this is in world frame, but normalized by the body's rotation and the arm's position around the body.

        :return: relative angle from each arm to the target, shape (envs, arms)
        :rtype: jnp.ndarray
        """
        num_arms = self.config.env.num_arms
        absolute_body_rotation = self.env_state.observations["disk_rotation"][:, jnp.newaxis, 2] # type: ignore -- the rotation around the z-axis (this is a scalar angle in radians) -- shape: (envs, 1)

        relative_arm_rotation  = jnp.array([
            i * (2 * jnp.pi / num_arms) for i in range(num_arms)
        ]) # relative to the body of the brittle star

        relative_arm_rotation = relative_arm_rotation[jnp.newaxis, :] # shape: (1, num_arms)

        absolute_body_position = self.env_state.observations["disk_position"][:, :2] # type: ignore -- shape: (envs, 2)
        direction_from_brittle_star_to_target = self.env_state.observations["unit_xy_direction_to_target"] # type: ignore -- shape: (envs, 2)
        distance_from_brittle_star_to_target = self.env_state.observations["xy_distance_to_target"] # type: ignore -- shape: (envs, 1)
        absolute_target_position = absolute_body_position + direction_from_brittle_star_to_target * distance_from_brittle_star_to_target # shape: (envs, 2)

        absolute_angle_body_to_target = jnp.arctan2( # arctan2(y, x) gives the angle between the positive x-axis and the point (x, y) in radians
            absolute_target_position[:, 1], # type: ignore -- shape: (envs,)
            absolute_target_position[:, 0]  # type: ignore -- shape: (envs,)
        ) % (2 * jnp.pi) # shape (envs,)

        absolute_angle_body_to_target = absolute_angle_body_to_target[:, jnp.newaxis] # shape: (envs, 1) is broadcastable to (envs, num_arms), while (envs,) is not

        relative_angle_body_to_target = (absolute_angle_body_to_target - absolute_body_rotation) % (2 * jnp.pi)

        relative_angle_arm_to_target_unnormalized = (relative_angle_body_to_target - relative_arm_rotation) % (2 * jnp.pi)
        relative_angle_arm_to_target = jnp.where(
            relative_angle_arm_to_target_unnormalized > jnp.pi,
            relative_angle_arm_to_target_unnormalized - 2 * jnp.pi,
            relative_angle_arm_to_target_unnormalized
        ) # return the signed angle in the range [-pi, pi]

        return relative_angle_arm_to_target[:, :, jnp.newaxis] # shape: (envs, arms, 1) - add extra dimension to get correct output shape for observations (num_envs, num_arms, obs_per_arm)
