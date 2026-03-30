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

from brittle_star_locomotion.config.config_loader import load_config
from brittle_star_locomotion.cpg.cpg import CPG, create_cpg_structure
from brittle_star_locomotion.cpg.solver import RK4Solver
from brittle_star_locomotion.gait.gait import map_cpg_to_brittle_star_actions, modulate_rowing_gait

logger = logging.getLogger(__name__)
config = load_config("configs/base_config.yaml")


class Environment:
    """
    Representation of the Brittle Star directed locomotion environment.
    
    This class integrates a Central Pattern Generator (CPG) with a MuJoCo-based 
    simulation (MJX) to coordinate multi-arm locomotion.
    """

    def __init__(self, observations: None | list[str] = None):
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

        self.num_arms = config.env.num_arms
        self.num_segments = config.env.num_segments_per_arm

        self.rng = jax.random.PRNGKey(seed=0)
        self.rng, self.reset_key = jax.random.split(self.rng)
        self.env_state = self.environment.reset(self.reset_key)

        self.weights = create_cpg_structure(self.num_arms * config.env.num_oscillators_per_segment)
        self.cpg = CPG(self.weights, RK4Solver(), config.cpg.dt)
        self.cpg_state = self.cpg.reset(self.reset_key)

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
        self.jit_env_step = jax.jit(self.environment.step)
        self.jit_env_reset = jax.jit(self.environment.reset)

    @functools.partial(jax.jit, static_argnums=(0,))
    def __step_compiled(self, env_state, cpg_state, masks, max_limit):
        """
        Pure JAX function to run simulation substeps via lax.scan.
        
        :param env_state: Current MJX environment state.
        :param cpg_state: Current CPG oscillator states.
        :param masks: Tuple of boolean masks for gait modulation.
        :param max_limit: Maximum joint excursion limit.
        :return: Tuple of (final_env_state, final_cpg_state, trajectory, summed_reward).
        """
        cpg_state = modulate_rowing_gait(cpg_state, *masks, max_joint_limit=max_limit)

        def _cpg_loop_body(_state, _):
            _next_state = self.cpg.step(_state)
            _action = map_cpg_to_brittle_star_actions(_next_state.outputs, config.env.num_arms, config.env.num_segments_per_arm)
            return _next_state, _action

        cpg_state, action_trajectory = jax.lax.scan(_cpg_loop_body, cpg_state, None, length=config.env.num_substeps_per_modulation)

        def _env_loop_body(_state, _action):
            _next_env_state = self.jit_env_step(_state, _action)

            return _next_env_state, (_next_env_state, _next_env_state.reward)

        final_env_state, (trajectory, rewards) = jax.lax.scan(_env_loop_body, env_state, action_trajectory)

        return final_env_state, cpg_state, trajectory, jnp.sum(rewards)
    

    def step(self, actions: jnp.ndarray) -> tuple:
        """
        Step the environment forward by applying CPG modulation and physics simulation.

        :param actions: Integer array of gait roles for each arm.
        :return: (env_state, reward, terminated, truncated)
        """
        # track position for progress-based reward
        prev_position = self.env_state.observations["disk_position"]

        masks = tuple(actions == i for i in range(5))
        new_env_state, new_cpg_state, _, _ = self.__step_compiled(
            self.env_state,
            self.cpg_state,
            masks,
            self.environment.action_space.high[0] * 0.5,  # type: ignore
        )

        self.env_state = new_env_state
        self.cpg_state = new_cpg_state

        current_position = self.env_state.observations["disk_position"]
        target_position = self.env_state.mj_model.body("target").pos
        
        # reward is distance reduction toward target
        previous_distance = jnp.linalg.norm(prev_position - target_position)
        current_distance = jnp.linalg.norm(current_position - target_position)
        reward = previous_distance - current_distance

        # bonus for reaching target (termination)
        reward += jnp.where(self.env_state.terminated, 10.0, 0.0)

        return self.env_state, reward, self.env_state.terminated, self.env_state.truncated
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def get_observations(self) -> jnp.ndarray:
        """
        Concatenates and returns the selected observation features per arm.

        :return: JAX array of shape (num_arms, observation_space_size).
        """
        obs_list = []

        for obs in self.observations:
            if obs == "angle_to_target":
                disk_xy = self.env_state.observations["disk_position"][:2] # type: ignore

                # 2. get arm root positions
                # reshape joint positions to (num_arms, num_segments, coordinates_per_joint)
                arm_pos = self.env_state.observations["joint_position"].reshape(self.num_arms, -1)
                arm_root_xy = arm_pos[:, :2] # take the first two coordinates (x, y)

                # 3. calculate vector from arm to disk (broadcasting (2,) against (5, 2))
                to_arm = disk_xy - arm_root_xy 
                
                # 4. normalize and compute dot product with target direction
                to_arm_unit = to_arm / (jnp.linalg.norm(to_arm, axis=1, keepdims=True) + 1e-8)
                to_target = self.env_state.observations["unit_xy_direction_to_target"] # (2,)
                
                # final dot product per arm: (5, 2) * (2,) -> (5, 1)
                val = jnp.sum(to_arm_unit * to_target[None, :], axis=1, keepdims=True) # type: ignore
                obs_list.append(val)

            elif obs == "arm_identification":
                obs_list.append(jnp.eye(self.num_arms))

            elif obs in self.state_space["central"]:
                # tile central observations for each arm
                val = jnp.tile(self.env_state.observations[obs], (self.num_arms, 1))
                obs_list.append(val.reshape(self.num_arms, -1))

            elif obs in self.state_space["individual_per_segment"] or obs in self.state_space["individual_per_arm"]:
                # reshape individual arm/segment data to (num_arms, -1)
                val = self.env_state.observations[obs].reshape(self.num_arms, -1)
                obs_list.append(val)

        return jnp.concatenate(obs_list, axis=-1)

    def run_iteration(self, action_values: jnp.ndarray, max_limit: float = 1.0):
        """Python wrapper to prepare masks and update instance state."""
        masks = tuple(action_values == i for i in range(5))

        new_env_state, new_cpg_state, trajectory, _ = self.__step_compiled(self.env_state, self.cpg_state, masks, max_limit)

        self.env_state = new_env_state
        self.cpg_state = new_cpg_state

        return trajectory

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
        """processes the trajectory into a video file using the environment's render logic."""
        logger.info(f"Rendering results to {output_path}")
        frames = []

        actual_steps = jax.tree_util.tree_leaves(trajectory)[0].shape[0]
        render_indices = range(0, actual_steps, config.env.render_every)

        for i in tqdm(render_indices, desc="Generating Video Frames"):
            step_state = jax.tree_util.tree_map(lambda x: x[i], trajectory)

            raw_frames = self.environment.render(step_state)
            if raw_frames is not None:
                processed_list = [np.asarray(f) for f in raw_frames]

                combined_frame = self.__post_render(processed_list)

                if combined_frame is not None:
                    frames.append(combined_frame)

        if frames:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            media.write_video(str(output_file), np.array(frames), fps=20)
            logger.info(f"Successfully saved video ({len(frames)} frames) to {output_path}")

    def show_video(self, video_path: str):
        """display the video in a notebook environment."""
        if Path(video_path).exists():
            media.show_video(media.read_video(video_path))

    def reset(self):  # TODO return type
        """Reset the environment

        :return: The new state.
        :rtype: BaseEnvState
        """
        self.reset_key, reset_key = jax.random.split(self.reset_key)
        self.cpg_state = self.cpg.reset(reset_key)
        self.env_state = self.jit_env_reset(self.reset_key)

        return self.env_state

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
                size += config.env.num_segments_per_arm * self.state_space["individual_per_segment"][obs]

            elif obs in self.state_space["individual_per_arm"]:
                size += self.state_space["individual_per_arm"][obs]

        return size
