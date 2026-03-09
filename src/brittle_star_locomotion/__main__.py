import jax
import jax.numpy as jnp
import numpy as np
import mediapy as media
from functools import partial

from brittle_star_locomotion.cpg.cpg import CPG
from brittle_star_locomotion.cpg.solver import RK4Solver
from brittle_star_locomotion.environment.environment import Environment
from brittle_star_locomotion.gait.gait import modulate_rowing_gait

NUM_ARMS: int = 5
NUM_SEGMENTS: int = 3
SIMULATION_TIME: float = 20.0
RENDER_EVERY: int = 5


def create_cpg_structure(num_osc: int) -> jnp.ndarray:
    """Creates the coupling weight matrix for the CPG network."""
    weights = jnp.zeros((num_osc, num_osc))
    ip_oscillators = jnp.arange(0, num_osc, 2)
    oop_oscillators = jnp.arange(1, num_osc, 2)

    weights = weights.at[ip_oscillators, oop_oscillators].set(1.0)
    next_ip = jnp.roll(ip_oscillators, shift=-1)
    weights = weights.at[ip_oscillators, next_ip].set(1.0)
    next_oop = jnp.roll(oop_oscillators, shift=-1)
    weights = weights.at[oop_oscillators, next_oop].set(1.0)

    weights = 5.0 * jnp.maximum(weights, weights.T)
    return weights


@partial(jax.jit, static_argnums=(0, 2, 3))
def run_jax_simulation(cpg, init_cpg_state, env_jax, num_steps):
    """Executes the entire CPG and Physics simulation loop using jax.lax.scan."""

    def single_step(carry, _):
        c_state, e_state = carry

        next_c_state = cpg.step(c_state)

        cpg_outputs_per_arm = next_c_state.outputs.reshape((NUM_ARMS, 2))
        actions = jnp.repeat(cpg_outputs_per_arm, NUM_SEGMENTS, axis=0).ravel()

        next_e_state = env_jax.step(e_state, actions)

        return (next_c_state, next_e_state), next_e_state

    init_e_state = env_jax.reset(jax.random.PRNGKey(0))
    _, trajectory = jax.lax.scan(single_step, (init_cpg_state, init_e_state), None, length=num_steps)
    return trajectory


def run_simulation():
    """Initializes configurations, runs the compiled simulation, and renders the result."""
    from biorobot.brittle_star.mjcf.arena.aquarium import AquariumArenaConfiguration
    from biorobot.brittle_star.environment.directed_locomotion.shared import BrittleStarDirectedLocomotionEnvironmentConfiguration

    arena_config = AquariumArenaConfiguration(size=(10, 5), attach_target=True, sand_ground_color=False)
    env_config = BrittleStarDirectedLocomotionEnvironmentConfiguration(
        render_mode="rgb_array",
        simulation_time=SIMULATION_TIME,
        render_size=(480, 640),
        camera_ids=(0, 1),
        time_scale=1,
        num_physics_steps_per_control_step=10,
    )

    env = Environment(
        num_arms=NUM_ARMS,
        num_segments_per_arm=NUM_SEGMENTS,
        arena_configuration=arena_config,
        environment_configuration=env_config,
    )

    num_osc = NUM_ARMS * 2
    weights = create_cpg_structure(num_osc)
    cpg = CPG(weights=weights, dt=env_config.control_timestep, solver=RK4Solver())
    cpg_state = cpg.reset(rng=jax.random.PRNGKey(0))

    cpg_state = cpg_state.replace(  # type: ignore
        omegas=jnp.pi / 2 * jnp.ones_like(cpg_state.omegas)
    )

    leading_idx = 0
    left_rowers = [(leading_idx - 1) % NUM_ARMS, (leading_idx - 2) % NUM_ARMS]
    right_rowers = [(leading_idx + 1) % NUM_ARMS, (leading_idx + 2) % NUM_ARMS]

    cpg_state = modulate_rowing_gait(
        cpg_state=cpg_state,
        leading_arms=[leading_idx],
        left_rowers=left_rowers,
        right_rowers=right_rowers,
        left_second=[left_rowers[1]],
        right_second=[right_rowers[1]],
        max_joint_limit=env.env.action_space.high[0] * 0.5,  # type: ignore
    )

    num_steps = int(SIMULATION_TIME / env_config.control_timestep)
    print("Starting JAX-native simulation...")

    trajectory = run_jax_simulation(cpg, cpg_state, env.env, num_steps)

    print("Rendering video from trajectory...")
    frames = []
    for i in range(0, num_steps, RENDER_EVERY):
        step_state = jax.tree_util.tree_map(lambda x: x[i], trajectory)
        rendered = np.asarray(env.env.render(step_state))
        combined = np.concatenate([rendered[0], rendered[1]], axis=1)
        frames.append(combined)

    if frames:
        media.write_video("out/modulated_rowing.mp4", np.array(frames))
        print("Video saved to out/modulated_rowing.mp4")

    env.env.close()


if __name__ == "__main__":
    # import os
    # import jax.profiler

    # jax_profile_dir = "/tmp/jax-prof"
    # os.makedirs(jax_profile_dir, exist_ok=True)

    # jax.profiler.start_trace(jax_profile_dir)
    run_simulation()
    # jax.profiler.stop_trace()
