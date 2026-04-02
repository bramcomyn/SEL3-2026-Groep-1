import functools
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import mediapy as media
import numpy as np
from biorobot.brittle_star.environment.directed_locomotion.dual import BrittleStarDirectedLocomotionEnvironment
from biorobot.brittle_star.environment.directed_locomotion.shared import BrittleStarDirectedLocomotionEnvironmentConfiguration
from biorobot.brittle_star.mjcf.arena.aquarium import AquariumArenaConfiguration, MJCFAquariumArena
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from biorobot.brittle_star.mjcf.morphology.specification.default import default_brittle_star_morphology_specification
from tqdm import tqdm

from brittle_star_locomotion.cpg.cpg import CPG, create_cpg_structure
from brittle_star_locomotion.cpg.solver import RK4Solver
from brittle_star_locomotion.gait.gait import map_cpg_to_brittle_star_actions, modulate_rowing_gait

logger = logging.getLogger(__name__)

class Environment:
    """
    Representation of the Brittle Star directed locomotion environment.
    
    This class integrates a Central Pattern Generator (CPG) with a MuJoCo-based 
    simulation (MJX) to coordinate multi-arm locomotion.
    """

    def __init__(
            self,
            config,
            observations: None | list[str] = None,
        ):
        self.amount_environments = config.rl.amount_environments # not pulling from config, because can be changed in __main__.py
        self.morphology_specification = default_brittle_star_morphology_specification(
            num_arms=config.env.num_arms, num_segments_per_arm=config.env.num_segments_per_arm, use_p_control=True, use_torque_control=False
        )

        self.arena_configuration = AquariumArenaConfiguration(
            size=(config.env.arena.size_x, config.env.arena.size_y), sand_ground_color=config.env.arena.sand_ground_color, attach_target=config.env.arena.attach_target, wall_height=config.env.arena.wall_height, wall_thickness=config.env.arena.wall_thickness
        )

        self.environment_configuration = BrittleStarDirectedLocomotionEnvironmentConfiguration(
            target_distance=config.env.target_distance,
            joint_randomization_noise_scale=0.0,
            render_mode="rgb_array",
            simulation_time=config.env.simulation_time,
            num_physics_steps_per_control_step=config.env.num_physics_steps_per_control_step,
            time_scale=config.env.time_scale,
            camera_ids=(0, 1),
            render_size=(config.env.render_size_x, config.env.render_size_y),
        )

        self.__create_environment()

        self.config = config
        self.num_arms = config.env.num_arms
        self.num_segments = config.env.num_segments_per_arm

        self.rng = jax.random.PRNGKey(seed=0)

        self.rng, reset_key, *self.sub_rngs = jnp.array(jax.random.split(self.rng, self.amount_environments + 2))

        self.weights = create_cpg_structure(config.env.num_oscillators_per_segment * config.env.num_arms)
        self.cpg = CPG(self.weights, RK4Solver(), config.cpg.dt)
        cpg_keys = jax.random.split(reset_key, self.amount_environments)
        self.cpg_state = jax.vmap(self.cpg.reset)(cpg_keys)
        self.env_state = self.environment.reset(reset_key)

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
                "arm_identification": self.num_arms,
            },
        }

        valid_keys = (
            list(self.state_space["central"].keys()) +
            list(self.state_space["individual_per_segment"].keys()) +
            list(self.state_space["individual_per_arm"].keys())
        )

        if observations is None:
            self.observations = valid_keys
        else:
            for obs in observations:
                assert obs in valid_keys, f"Observation {obs} not in state space."
            self.observations = observations

        self.observation_space_size = self.get_observation_size()

        self.jit_env_step = jax.jit(jax.vmap(self.environment.step))
        self.jit_env_reset = jax.jit(jax.vmap(self.environment.reset))

    @functools.partial(jax.jit, static_argnums=(0,))
    def __step_compiled(
        self, 
        env_state, 
        cpg_state, 
        masks: jnp.ndarray, # shape (5 x 5)
        max_limit
    ):  
        """
        Processes one environment's worth of physics and control.
        
        :param env_state: Current MJX environment state.
        :param cpg_state: Current CPG oscillator states.
        :param masks: Tuple of boolean masks for gait modulation.
        :param max_limit: Maximum joint excursion limit.

        :return: Tuple of (final_env_state, final_cpg_state, trajectory, summed_reward).
        """
        cpg_state = modulate_rowing_gait(cpg_state, masks, max_joint_limit=max_limit)

        def _cpg_loop_body(_state, _):
            _next_state = self.cpg.step(_state)
            _action = map_cpg_to_brittle_star_actions(_next_state.outputs, self.config.env.num_arms, self.config.env.num_segments_per_arm)
            return _next_state, _action

        cpg_state, action_trajectory = jax.lax.scan(_cpg_loop_body, cpg_state, None, length=self.config.env.num_substeps_per_modulation)

        def _env_loop_body(_state, _action):
            _next_env_state = self.environment.step(_state, _action)
            return _next_env_state, (_next_env_state, _next_env_state.reward)

        final_env_state, (trajectory, rewards) = jax.lax.scan(_env_loop_body, env_state, action_trajectory)

        return final_env_state, cpg_state, trajectory, jnp.sum(rewards)

    def __create_environment(self):
        """create an environment configuration based on the self.morphology_specification, self.arena_configuration and self.environment_configuration"""
        self.morphology = MJCFBrittleStarMorphology(self.morphology_specification)
        self.arena = MJCFAquariumArena(self.arena_configuration)
        self.environment = BrittleStarDirectedLocomotionEnvironment.from_morphology_and_arena(
            self.morphology, self.arena, self.environment_configuration, "MJX"
        )

    def __post_render(self, render_output: list[np.ndarray]) -> np.ndarray | None:
        """converts list of camera arrays into a single stitched array."""
        if render_output is None or len(render_output) == 0:
            return None

        num_cameras = len(self.environment_configuration.camera_ids)

        # If we have multiple cameras, stitch them side-by-side (axis=1)
        if num_cameras > 1:
            processed_frame = np.concatenate(render_output, axis=1)
        else:
            processed_frame = render_output[0]

        return processed_frame

    def render_video(self, trajectory, output_path: str = "out/test-video.mp4"):
        """Processes trajectory for ALL environments and stacks them vertically."""
        logger.info(f"Rendering all environments to {output_path}")
        frames = []

        # dimensions: (n_envs, steps, ...)
        first_leaf = jax.tree_util.tree_leaves(trajectory)[0]
        total_steps = first_leaf.shape[1]
        num_steps = first_leaf.shape[0]
        render_indices = range(0, total_steps, self.config.env.render_every)

        for i in tqdm(render_indices, desc="Generating Video Frames"):
            env_frames_for_this_step = []
            
            for e in range(num_steps):
                # extract state for step 'e' and time 'i'
                step_state = jax.tree_util.tree_map(lambda x: x[e, i], trajectory)
                raw_frames = self.environment.render(step_state)
                
                if raw_frames is not None:
                    processed_list = [np.asarray(f) for f in raw_frames]
                    combined_camera_view = self.__post_render(processed_list)
                    env_frames_for_this_step.append(combined_camera_view)

            frames.extend(env_frames_for_this_step)

        if frames:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            media.write_video(str(output_file), np.array(frames), fps=20)
            logger.info(f"Saved multi-env video ({len(frames)} frames) to {output_path}")

    def show_video(self, video_path: str):
        """display the video in a notebook environment."""
        if Path(video_path).exists():
            media.show_video(media.read_video(video_path))

    def step(self, actions: jnp.ndarray):
        """Step in the environment.

        :param actions: Actions to take in the environment, one float for each actuator.
        :type jnp.ndarray: actions (n_envs, n_agents)

        :return: The new state, reward, termination status, truncation status, and info.
        :rtype: tuple
        """
        previous_distance = self.env_state.observations["xy_distance_to_target"] # (envs, 1)

        masks = actions[:, jnp.newaxis, :] == jnp.arange(5)[jnp.newaxis, :, jnp.newaxis]

        vmapped_step = jax.vmap(self.__step_compiled, in_axes=(0, 0, 0, None))
        new_env_state, new_cpg_state, trajectory, _ = vmapped_step(
            self.env_state,
            self.cpg_state,
            masks,
            self.environment.action_space.high[0] * 0.5,  # type: ignore
        )

        self.env_state = new_env_state
        self.cpg_state = new_cpg_state

        current_distance = self.env_state.observations["xy_distance_to_target"] # (envs, 1)

        reward = (previous_distance - current_distance)      # (envs, 1)
        reward += 10.0 * self.env_state.terminated[:, jnp.newaxis]  # (envs, 1)

        return self.env_state, reward, self.env_state.terminated, self.env_state.truncated, trajectory

    def reset(self):
        """Reset both the MJX environment and the CPG controllers."""
        # self.sub_rngs should be shape (num_envs, 2)
        self.env_state = self.jit_env_reset(jnp.array(self.sub_rngs))

        # Generate keys for each environment's CPG
        self.rng, cpg_reset_key = jax.random.split(self.rng)
        cpg_keys = jax.random.split(cpg_reset_key, self.amount_environments)
        self.cpg_state = jax.vmap(self.cpg.reset)(cpg_keys)

        return self.env_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_observations(self) -> jnp.ndarray:
        # oversight: every arm needs to know where it is and where the target is.
        # broadcasts central data (like disk velocity) to all arms.
        # resulting shape: (num_envs, num_arms, total_obs_per_arm)
        obs_list = []
        num_envs = self.amount_environments
        num_arms = self.config.env.num_arms

        for obs in self.observations:
            if obs not in self.derived_states:
                if obs in self.state_space["central"]:
                    data = self.env_state.observations[obs][:, jnp.newaxis, :] # type: ignore       # (envs, 1, central_dim) 
                    obs_list.append(jnp.broadcast_to(data, (num_envs, num_arms, data.shape[-1])))   # (envs, arms, central_dim)

                elif obs in self.state_space["individual_per_segment"]:
                    data = self.env_state.observations[obs]                 # (envs, arms * segments * dim)
                    obs_list.append(data.reshape(num_envs, num_arms, -1))   # (envs, arms, segments * dim)

                elif obs in self.state_space["individual_per_arm"]:
                    data = self.env_state.observations[obs]                 # (envs, arms * dim) 
                    obs_list.append(data.reshape(num_envs, num_arms, -1))   # (envs, arms, dim)

            else:
                if obs == "angle_to_target":
                    joint_position =  self.env_state.observations["joint_position"]      # (envs, arms * segments * dim)
                    arm_pos = joint_position.reshape(num_envs, num_arms, -1)[:, :, :2]   # (envs, arms, 2)

                    disk_position = self.env_state.observations["disk_position"]         # (envs, 3)
                    disk_position = disk_position[:, jnp.newaxis, :2] # type: ignore     # (envs, 1, 2)

                    to_arm = arm_pos - disk_position    # (envs, arms, 2)
                    to_arm = to_arm / jnp.linalg.norm(to_arm, axis=-1, keepdims=True)   # (envs, arms, 2)

                    to_target = self.env_state.observations["unit_xy_direction_to_target"][:, jnp.newaxis, :] # type: ignore    # (envs, 1, 2)

                    # Cosine similarity (dot product of unit vectors)
                    angle_obs = jnp.sum(to_arm * to_target, axis=-1, keepdims=True)     # (envs, arms, 1)
                    obs_list.append(angle_obs)

        return jnp.concatenate(obs_list, axis=-1)

    def get_observation_size(self) -> int:
        """Get the size of the observation space.

        :return: The size of the observation space.
        :rtype: int
        """
        size = 0

        for obs in self.observations:
            if obs in self.state_space["central"]:
                size += self.state_space["central"][obs]

            elif obs in self.state_space["individual_per_segment"]:
                size += self.config.env.num_segments_per_arm * self.state_space["individual_per_segment"][obs]

            elif obs in self.state_space["individual_per_arm"]:
                size += self.state_space["individual_per_arm"][obs]

        return size
