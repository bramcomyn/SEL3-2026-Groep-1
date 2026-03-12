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
from brittle_star_locomotion.optimisation.iql import IQL
from flax import nnx
from tqdm import tqdm

NUM_ARMS: int = 5
NUM_SEGMENTS: int = 3
RENDER_EVERY: int = 5

logger = logging.getLogger(__name__)


def run_experiment(simulation_time: float, should_render: bool):
    def loss_fn(target, predicted):
        return jnp.mean((predicted - target) ** 2)

    arena_config = AquariumArenaConfiguration(size=(10, 5), attach_target=True, sand_ground_color=False)
    env_config = BrittleStarDirectedLocomotionEnvironmentConfiguration(
        render_mode="rgb_array",
        simulation_time=simulation_time,
        render_size=(480, 640),
        camera_ids=(0, 1),
        time_scale=1,
        num_physics_steps_per_control_step=10,
    )

    control = CPGControl(env_config.control_timestep, 0)

    env = Environment(
        num_arms=NUM_ARMS, num_segments_per_arm=NUM_SEGMENTS, arena_configuration=arena_config, environment_configuration=env_config, control=control
    )

    iql = IQL(optax.adam(1e-3), loss_fn, 5, env)

    iql.train()


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
    )

    num_osc = NUM_ARMS * 2
    weights = create_cpg_structure(num_osc)
    cpg = CPG(weights=weights, dt=env_config.control_timestep, solver=RK4Solver())

    return env, cpg, env_config


def initialize_gait(cpg: CPG, env: Environment) -> Any:
    """Sets up the modulated rowing gait state."""
    logger.info("Initializing CPG state with Modulated Rowing Gait")
    cpg_state = cpg.reset(rng=jax.random.PRNGKey(0))

    # Set base frequency
    cpg_state = cpg_state.replace(omegas=jnp.pi / 2 * jnp.ones_like(cpg_state.omegas))  # type: ignore

    # Define rowing roles
    leading_idx = 0
    left_rowers = [(leading_idx - 1) % NUM_ARMS, (leading_idx - 2) % NUM_ARMS]
    right_rowers = [(leading_idx + 1) % NUM_ARMS, (leading_idx + 2) % NUM_ARMS]

    return modulate_rowing_gait(
        cpg_state=cpg_state,
        leading_arms=[leading_idx],
        left_rowers=left_rowers,
        right_rowers=right_rowers,
        left_second=[left_rowers[1]],
        right_second=[right_rowers[1]],
        max_joint_limit=env.env.action_space.high[0] * 0.5,  # type: ignore
    )


@partial(jax.jit, static_argnums=(0, 2, 3))
def run_jax_simulation(cpg, init_cpg_state, env_jax, num_steps):
    """The JIT-compiled simulation loop."""
    logger.debug(f"JAX Tracing: Compiling simulation for {num_steps} steps")

    def single_step(carry, _):
        c_state, e_state = carry
        next_c_state = cpg.step(c_state)

        # Map CPG outputs to robot actuators
        cpg_outputs_per_arm = next_c_state.outputs.reshape((NUM_ARMS, 2))
        actions = jnp.repeat(cpg_outputs_per_arm, NUM_SEGMENTS, axis=0).ravel()

        next_e_state = env_jax.step(e_state, actions)
        return (next_c_state, next_e_state), next_e_state

    init_e_state = env_jax.reset(jax.random.PRNGKey(0))
    _, trajectory = jax.lax.scan(single_step, (init_cpg_state, init_e_state), None, length=num_steps)
    return trajectory


def render_video(env: Environment, trajectory: Any, num_steps: int, output_path: str):
    """Processes the trajectory into a video file."""
    logger.info(f"Rendering results to {output_path}")
    frames = []

    render_indices = range(0, num_steps, RENDER_EVERY)
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


# def run_experiment(simulation_time: float, should_render: bool):
#     """Top-level entry point for a single experiment run."""
#     env, cpg, env_config = setup_simulation_objects(simulation_time)
#     cpg_state = initialize_gait(cpg, env)

#     num_steps = int(simulation_time / env_config.control_timestep)

#     logger.info("Executing JAX simulation...")
#     trajectory = run_jax_simulation(cpg, cpg_state, env.env, num_steps)

#     # Block until JAX is finished to ensure timing is accurate
#     jax.block_until_ready(trajectory)
#     logger.info("Simulation execution complete.")

#     if should_render:
#         render_video(env, trajectory, num_steps, "out/modulated_rowing.mp4")

#     env.env.close()
#     logger.info("Environment closed.")
