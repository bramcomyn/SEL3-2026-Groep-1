import jax
import jax.numpy as jnp
import numpy as np
import mediapy as media

from biorobot.brittle_star.environment.directed_locomotion.shared import BrittleStarDirectedLocomotionEnvironmentConfiguration
from biorobot.brittle_star.mjcf.arena.aquarium import AquariumArenaConfiguration

from brittle_star_locomotion.cpg.cpg import CPG
from brittle_star_locomotion.cpg.solver import RK4Solver
from brittle_star_locomotion.environment.environment import Environment

num_arms = 5
num_segments = 4
num_osc = num_arms * num_segments * 2
dt = 0.01

arena_configuration = AquariumArenaConfiguration(size=(10, 5), sand_ground_color=False, attach_target=False, wall_height=1.5, wall_thickness=0.1)
environment_configuration = BrittleStarDirectedLocomotionEnvironmentConfiguration(render_mode="rgb_array", simulation_time=5, camera_ids=[1])

env = Environment(
    num_arms=num_arms,
    num_segments_per_arm=num_segments,
    arena_configuration=arena_configuration,
    environment_configuration=environment_configuration,
    backend="MJX",
)


def create_cpg_structure():
    weights = jnp.zeros((num_osc, num_osc))

    for i in range(0, num_osc, 2):
        weights = weights.at[i, i + 1].set(1.0)
        weights = weights.at[i + 1, i].set(1.0)

    for arm in range(num_arms):
        for s in range(num_segments - 1):
            curr_base = (arm * num_segments + s) * 2
            next_base = (arm * num_segments + s + 1) * 2
            weights = weights.at[curr_base, next_base].set(10.0)
            weights = weights.at[next_base, curr_base].set(10.0)
            weights = weights.at[curr_base + 1, next_base + 1].set(1.0)
            weights = weights.at[next_base + 1, curr_base + 1].set(1.0)

    return weights


def modulate_cpg(cpg_state, leading_arm_index, max_joint_limit):
    left_rowers = [(leading_arm_index - 1) % 5, (leading_arm_index - 2) % 5]
    right_rowers = [(leading_arm_index + 1) % 5, (leading_arm_index + 2) % 5]

    R = jnp.zeros(num_osc)
    X = jnp.zeros(num_osc)
    rhos = jnp.zeros((num_osc, num_osc))
    omegas = 2 * jnp.pi * jnp.ones(num_osc)

    segment_delay = 0.6

    lead_base = leading_arm_index * num_segments * 2
    for s in range(num_segments):
        idx = lead_base + s * 2
        X = X.at[idx + 1].set(max_joint_limit)

    for arm_idx in left_rowers:
        arm_base = arm_idx * num_segments * 2
        for s in range(num_segments):
            ip, oop = arm_base + s * 2, arm_base + s * 2 + 1
            amp = max_joint_limit * (1.0 - (s * 0.12))
            R = R.at[ip].set(amp)
            R = R.at[oop].set(amp * 0.5)
            rhos = rhos.at[ip, oop].set(jnp.pi / 2)
            rhos = rhos.at[oop, ip].set(-jnp.pi / 2)
            if s < num_segments - 1:
                next_ip = arm_base + (s + 1) * 2
                next_oop = arm_base + (s + 1) * 2 + 1
                rhos = rhos.at[ip, next_ip].set(segment_delay)
                rhos = rhos.at[next_ip, ip].set(-segment_delay)
                rhos = rhos.at[oop, next_oop].set(segment_delay)
                rhos = rhos.at[next_oop, oop].set(-segment_delay)

    for arm_idx in right_rowers:
        arm_base = arm_idx * num_segments * 2
        for s in range(num_segments):
            ip, oop = arm_base + s * 2, arm_base + s * 2 + 1
            amp = max_joint_limit * (1.0 - (s * 0.12))
            R = R.at[ip].set(amp)
            R = R.at[oop].set(amp * 0.5)
            rhos = rhos.at[ip, oop].set(-jnp.pi / 2)
            rhos = rhos.at[oop, ip].set(jnp.pi / 2)
            if s < num_segments - 1:
                next_ip = arm_base + (s + 1) * 2
                next_oop = arm_base + (s + 1) * 2 + 1
                rhos = rhos.at[ip, next_ip].set(segment_delay)
                rhos = rhos.at[next_ip, ip].set(-segment_delay)
                rhos = rhos.at[oop, next_oop].set(segment_delay)
                rhos = rhos.at[next_oop, oop].set(-segment_delay)

    return cpg_state.replace(R=R, X=X, rhos=rhos, omegas=omegas)


weights = create_cpg_structure()
cpg = CPG(weights=weights, dt=dt, solver=RK4Solver())
cpg_state = cpg.reset(rng=jax.random.PRNGKey(0))
cpg_state = modulate_cpg(cpg_state, leading_arm_index=0, max_joint_limit=4)

jit_env_step = jax.jit(env.step)
jit_env_reset = jax.jit(env.reset)
env_state = jit_env_reset(jax.random.PRNGKey(42))

frames = []
render_every = 5

for i in range(int(environment_configuration.simulation_time / dt)):
    cpg_state = cpg.step(cpg_state)
    actions = cpg_state.outputs
    env_state = jit_env_step(env_state, actions)
    if env_state.terminated or env_state.truncated:
        break
    if i % render_every == 0:
        frames.append(np.asarray(env.env.render(env_state)).squeeze())

if frames:
    media.write_video("out/modulated_rowing.mp4", np.array(frames), fps=1.0 / (dt * render_every))

env.env.close()
