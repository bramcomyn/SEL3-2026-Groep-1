import copy
import logging
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mediapy as media
import numpy as np
import optax
from biorobot.brittle_star.environment.directed_locomotion.shared import BrittleStarDirectedLocomotionEnvironmentConfiguration
from biorobot.brittle_star.mjcf.arena.aquarium import AquariumArenaConfiguration
from brittle_star_locomotion.control.control import CPGControl
from brittle_star_locomotion.cpg.cpg import CPG, create_cpg_structure
from brittle_star_locomotion.cpg.solver import RK4Solver
from brittle_star_locomotion.environment.environment import Environment
from brittle_star_locomotion.gait.gait import modulate_rowing_gait
from brittle_star_locomotion.nn.checkpointing import load_checkpoint, save_checkpoint
from brittle_star_locomotion.nn.q_network import QNetwork
from brittle_star_locomotion.optimisation.iql import IQL
from flax import nnx
from tqdm import tqdm

NUM_ARMS: int = 5
NUM_SEGMENTS: int = 3
RENDER_EVERY: int = 5

logger = logging.getLogger(__name__)


def run_experiment(simulation_time: float):
    arena_config = AquariumArenaConfiguration(size=(10, 5), attach_target=True, sand_ground_color=False)
    env_config = BrittleStarDirectedLocomotionEnvironmentConfiguration(
        render_mode="rgb_array",
        simulation_time=simulation_time,
        render_size=(480, 640),
        camera_ids=(0, 1),
        time_scale=1,
        num_physics_steps_per_control_step=10,
    )

    control = CPGControl(env_config.control_timestep)
    env = Environment(
        num_arms=NUM_ARMS,
        num_segments_per_arm=NUM_SEGMENTS,
        arena_configuration=arena_config,
        environment_configuration=env_config,
        control=control,
        # observations=["disk_position"],
    )

    iql = IQL(optimizer=optax.adam(1e-2), n_agents=5, env=env)
    iql.train()
    save_checkpoint(iql.value_network, "test_checkpoint")


def visualize_agent(checkpoint: str, simulation_time: float):
    """
    Run a deterministic simulation using a trained QNetwork and record the trajectory
    for rendering, avoiding TraceContextError.
    """
    # Setup environment
    env, _, env_config = setup_simulation_objects(simulation_time)
    observation_size = env.get_observation_size()
    n_agents = 5

    # Load the trained network
    network = load_checkpoint(lambda: QNetwork(observation_size, n_agents, nnx.Rngs(0)), checkpoint)

    num_steps = int(simulation_time / env_config.control_timestep)
    trajectory = [env.state]

    for _ in tqdm(range(num_steps), desc="episode"):  # TODO num_steps
        obs = env.get_observations()

        # print(obs)

        q_values = network(obs)
        actions = jnp.argmax(q_values, axis=1)
        # print(actions)
        env.step(actions)  # TODO: can be overruled, see control step
        trajectory.append(env.state)

    # This turns a list of PyTrees into one PyTree of stacked arrays
    stacked_trajectory = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *trajectory)

    # Pass the stacked version, and remove the weird [0, trajectory] wrapping
    render_video(env, stacked_trajectory, len(trajectory), "out/test_checkpoint_video.mp4")


def setup_simulation_objects(simulation_time: float) -> tuple[Environment, CPG, Any]:
    """Initializes the MuJoCo environment and the CPG controller."""
    if simulation_time <= 0:
        raise ValueError("simulation_time must be positive")

    logger.debug(f"Configuring Aquarium Arena (Size: 10x5)")
    arena_config = AquariumArenaConfiguration(size=(10, 5), attach_target=True, sand_ground_color=False)

    env_config = BrittleStarDirectedLocomotionEnvironmentConfiguration(
        render_mode="rgb_array",
        simulation_time=simulation_time,
        render_size=(480, 640),
        camera_ids=(0, 1),
        time_scale=1,
        num_physics_steps_per_control_step=10,
    )

    logger.info(f"Initializing Environment: {NUM_ARMS} arms, {NUM_SEGMENTS} segments/arm")
    env = Environment(
        num_arms=NUM_ARMS,
        num_segments_per_arm=NUM_SEGMENTS,
        arena_configuration=arena_config,
        environment_configuration=env_config,
        control=CPGControl(env_config.control_timestep),
        # observations=["disk_position"],
    )

    num_osc = NUM_ARMS * 2
    weights = create_cpg_structure(num_osc)
    cpg = CPG(weights=weights, dt=env_config.control_timestep, solver=RK4Solver())

    return env, cpg, env_config


def render_video(env: Environment, trajectory: Any, num_steps: int, output_path: str):
    """Processes the trajectory into a video file."""
    logger.info(f"Rendering results to {output_path}")
    frames = []

    actual_steps = jax.tree_util.tree_leaves(trajectory)[0].shape[0]
    render_indices = range(0, actual_steps, RENDER_EVERY)
    for i in tqdm(render_indices, desc="Generating Video Frames"):
        step_state = jax.tree_util.tree_map(lambda x: x[i], trajectory)
        rendered = np.asarray(env.env.render(step_state))
        # Stitch camera views side-by-side
        combined = np.concatenate([rendered[0], rendered[1]], axis=1)
        frames.append(combined)

    if frames:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        media.write_video(output_path, np.array(frames))
        logger.info(f"Successfully saved video ({len(frames)} frames)")
