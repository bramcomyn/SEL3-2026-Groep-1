import jax
import jax.numpy as jnp
from brittle_star_locomotion.cpg.cpg import CPGState


def get_oscillator_indices_for_arm(arm_index: int) -> tuple[int, int]:
    """Returns (in-plane, out-of-plane) oscillator indices for a given arm."""
    return arm_index * 2, arm_index * 2 + 1


@jax.jit
def modulate_rowing_gait(
    cpg_state: CPGState,
    leading_arms: list[int],
    left_rowers: list[int],
    right_rowers: list[int],
    left_second: list[int],
    right_second: list[int],
    max_joint_limit: float,
) -> CPGState:
    """Modulates CPG parameters to produce a rowing gait.

    In rowing, the leading arm points forward while the other arms move
    symmetrically with specific phase offsets to generate thrust.
    """
    R = jnp.zeros_like(cpg_state.R)
    X = jnp.zeros_like(cpg_state.X)
    rhos = jnp.zeros_like(cpg_state.rhos)

    for lead_idx in leading_arms:
        _, oop_osc = get_oscillator_indices_for_arm(lead_idx)
        X = X.at[oop_osc].set(max_joint_limit)

    def apply_rower(arm_indices: list[int], bias_val: float, R_arr: jnp.ndarray, rhos_arr: jnp.ndarray):
        for idx in arm_indices:
            ip, oop = get_oscillator_indices_for_arm(idx)
            R_arr = R_arr.at[ip].set(max_joint_limit)
            R_arr = R_arr.at[oop].set(max_joint_limit)
            rhos_arr = rhos_arr.at[ip, oop].set(bias_val)
            rhos_arr = rhos_arr.at[oop, ip].set(-bias_val)
        return R_arr, rhos_arr

    R, rhos = apply_rower(left_rowers, jnp.pi / 2, R, rhos)
    R, rhos = apply_rower(right_rowers, -jnp.pi / 2, R, rhos)

    for l_idx, r_idx in zip(left_second, right_second):
        l_ip, _ = get_oscillator_indices_for_arm(l_idx)
        r_ip, _ = get_oscillator_indices_for_arm(r_idx)
        rhos = rhos.at[l_ip, r_ip].set(jnp.pi)
        rhos = rhos.at[r_ip, l_ip].set(-jnp.pi)

    return cpg_state.replace(R=R, X=X, rhos=rhos)  # type: ignore


def map_cpg_to_brittle_star_actions(outputs: jnp.ndarray, num_arms: int, num_segments: int) -> jnp.ndarray:
    """Broadcats 2 oscillators per arm across all segments of that arm."""
    # shape: (num_arms, 2)
    cpg_outputs_per_arm = outputs.reshape((num_arms, 2))
    # shape: (num_arms * num_segments, 2)
    cpg_outputs_per_segment = cpg_outputs_per_arm.repeat(num_segments, axis=0)
    return cpg_outputs_per_segment.flatten()
