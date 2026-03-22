import jax
import logging
import functools

import jax.numpy as jnp
import mediapy as media
import numpy as np

from biorobot.brittle_star.mjcf.morphology.specification.default import default_brittle_star_morphology_specification
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from biorobot.brittle_star.mjcf.arena.aquarium import AquariumArenaConfiguration, MJCFAquariumArena
from biorobot.brittle_star.environment.directed_locomotion.shared import BrittleStarDirectedLocomotionEnvironmentConfiguration
from biorobot.brittle_star.environment.directed_locomotion.dual import BrittleStarDirectedLocomotionEnvironment

from brittle_star_locomotion.cpg.solver import RK4Solver
from brittle_star_locomotion.cpg.cpg import create_cpg_structure, CPG
from brittle_star_locomotion.gait.gait import map_cpg_to_brittle_star_actions, modulate_rowing_gait

from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

NUM_ARMS = 5
NUM_OSCILLATORS = NUM_ARMS * 2
NUM_SEGMENTS_PER_ARM = 3
SIMULATION_TIME = 20
DT = 0.01
SEED=42
TARGET_DISTANCE = 1.2
RENDER_EVERY = 5
NUM_SUBSTEPS_PER_MODULATION=50

class Environment:
    """representation of the brittle star directed locomotion environment"""

    def __init__(self, observations: None | list[str]=None):
        self.morphology_specification = default_brittle_star_morphology_specification(
            num_arms=NUM_ARMS,
            num_segments_per_arm=NUM_SEGMENTS_PER_ARM,
            use_p_control=True,
            use_torque_control=False
        )

        self.arena_configuration = AquariumArenaConfiguration(
            size=(15, 15),
            sand_ground_color=False,
            attach_target=True,
            wall_height=1.5,
            wall_thickness=0.1
        )

        self.environment_configuration = BrittleStarDirectedLocomotionEnvironmentConfiguration(
            target_distance=TARGET_DISTANCE,
            joint_randomization_noise_scale=0.0,
            render_mode="rgb_array",
            simulation_time=SIMULATION_TIME,
            num_physics_steps_per_control_step=10,
            time_scale=1,
            camera_ids=(0, 1),
            render_size=(480, 640),
        )

        self.__create_environment()

        self.rng            = jax.random.PRNGKey(seed=SEED)
        self.rng, reset_key = jax.random.split(self.rng)
        self.weights        = create_cpg_structure(NUM_OSCILLATORS)
        self.cpg            = CPG(self.weights, RK4Solver(), DT)
        self.cpg_state      = self.cpg.reset(reset_key)
        self.env_state      = self.environment.reset(reset_key)
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

            # Derived observations
            "angle_to_target": 1
        }
        self.derived_state = ["angle_to_target"]

        if observations is None:
            self.observations = list(self.state_space.keys()) + self.derived_state
        else:
            for obs in observations:
                assert obs in self.state_space, f"Observation {obs} not in state space {self.state_space}"
            self.observations = observations

        self.jit_env_step = jax.jit(self.environment.step)
        self.jit_env_reset = jax.jit(self.environment.reset)


    @functools.partial(jax.jit, static_argnums=(0,))
    def __step_compiled(self, env_state, cpg_state, masks, max_limit):
        """Pure JAX function to run simulation substeps."""
        cpg_state = modulate_rowing_gait(cpg_state, *masks, max_joint_limit=max_limit)

        def _cpg_loop_body(_state, _):
            _next_state = self.cpg.step(_state)
            _action = map_cpg_to_brittle_star_actions(_next_state.outputs, NUM_ARMS, NUM_SEGMENTS_PER_ARM)
            return _next_state, _action

        cpg_state, action_trajectory = jax.lax.scan(
            _cpg_loop_body, cpg_state, None, length=NUM_SUBSTEPS_PER_MODULATION
        )

        def _env_loop_body(_state, _action):
            _next_env_state = self.jit_env_step(_state, _action)
            return _next_env_state, (_next_env_state, _next_env_state.reward)

        final_env_state, (trajectory, rewards) = jax.lax.scan(
            _env_loop_body, env_state, action_trajectory
        )

        return final_env_state, cpg_state, trajectory, jnp.sum(rewards)
    
    def run_iteration(self, action_values: jnp.ndarray, max_limit: float = 1.0):
        """Python wrapper to prepare masks and update instance state."""
        masks = tuple(action_values == i for i in range(5))

        new_env_state, new_cpg_state, trajectory, _ = self.__step_compiled(
            self.env_state, 
            self.cpg_state, 
            masks, 
            max_limit
        )
        
        self.env_state = new_env_state
        self.cpg_state = new_cpg_state
        
        return trajectory

    def __create_environment(self):
        """create an environment configuration based on the self.morphology_specification, self.arena_configuration and self.environment_configuration"""
        self.morphology  = MJCFBrittleStarMorphology(self.morphology_specification)
        self.arena       = MJCFAquariumArena(self.arena_configuration)
        self.environment = BrittleStarDirectedLocomotionEnvironment.from_morphology_and_arena(
            self.morphology,
            self.arena,
            self.environment_configuration,
            "MJX"
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

    def render_video(self, trajectory, output_path: str="out/test-video.mp4"):
        """processes the trajectory into a video file using the environment's render logic."""
        logger.info(f"Rendering results to {output_path}")
        frames = []

        actual_steps = jax.tree_util.tree_leaves(trajectory)[0].shape[0]
        render_indices = range(0, actual_steps, RENDER_EVERY)

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

    def step(self, actions: jnp.ndarray):  # TODO return type
        """Step in the environment.

        :param actions: Actions to take in the environment, one float for each actuator.
        :type jnp.ndarray: actions

        :return: The new state, reward, termination status, truncation status, and info.
        :rtype: tuple
        """        
        masks = tuple(actions == i for i in range(5))
        new_env_state, new_cpg_state, _, summed_reward = self.__step_compiled(
            self.env_state, self.cpg_state, masks, self.environment.action_space.high[0] * 0.5 # type: ignore
        )
        
        self.env_state = new_env_state
        self.cpg_state = new_cpg_state
        
        return self.env_state, summed_reward, self.env_state.terminated, self.env_state.truncated

    def reset(self):  # TODO return type
        """Reset the environment

        :return: The new state.
        :rtype: BaseEnvState
        """
        self.env_state = self.jit_env_reset(self.rng)
        return self.env_state

    def get_observations(self) -> jnp.ndarray:
        """Get the observations from the environment per agent.

        For each agent, this returns the selected observations from `observations` in the constructor.
        Individual observations are concatenated into a single array.
        Observations are then stacked along a new dimension.

        :return: The selected observations per agent with shape (num_agents, observation_space_shape).
        :rtype: jnp.ndarray
        """
        observation_space_shape = self.get_observation_size()
        observations = jnp.zeros((NUM_ARMS, observation_space_shape))

        for arm in range(NUM_ARMS):
            i = 0
            for obs in self.observations:
                for obs_idx in range(NUM_SEGMENTS_PER_ARM * self.state_space[obs]):

                    if obs in self.env_state.observations:
                        observations = observations.at[arm, i].set(self.env_state.observations[obs][arm * self.state_space[obs] + obs_idx])  # type: ignore

                    elif obs in self.derived_state:
                        if obs == "angle_to_target":
                            
                            to_target_unit_vec2 = self.env_state.observations["unit_xy_direction_to_target"][0 : 2]  # type: ignore
                                                        
                            to_arm_vec2 = self.env_state.observations["joint_position"][arm * 2 : arm * 2 + 2]  # type: ignore
                            to_arm_unit_vec2 = to_arm_vec2 / (jnp.linalg.norm(to_arm_vec2) + 1e-8)

                            angle = jnp.dot(to_target_unit_vec2, to_arm_unit_vec2)

                            observations = observations.at[arm, i].set(angle)

                    i += 1

            # observations = observations.at[arm, -1].set(arm)
        #     agent_one_hot_start = -NUM_ARMS  # start index of the one-hot section
        #     observations = observations.at[arm, agent_one_hot_start:].set(0.0)
        #     observations = observations.at[arm, agent_one_hot_start + arm].set(1.0)

        # # Normalize
        # mean = jnp.mean(observations, axis=0)  # mean per feature
        # std = jnp.std(observations, axis=0)
        # observations = (observations - mean) / (std + 1e-8)

        return observations

    def get_observation_size(self) -> int:
        """Get the size of the observation space.

        :return: The size of the observation space.
        :rtype: int
        """
        return NUM_SEGMENTS_PER_ARM * sum(self.state_space[obs] for obs in self.observations) # + NUM_ARMS
    