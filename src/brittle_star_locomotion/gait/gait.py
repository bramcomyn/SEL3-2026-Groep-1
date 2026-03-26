import jax
import jax.numpy as jnp
from brittle_star_locomotion.cpg.cpg import CPGState


def get_oscillator_indices_for_arm(arm_index):
    """Returns (in-plane, out-of-plane) oscillator indices."""
    return arm_index * 2, arm_index * 2 + 1


@jax.jit
def modulate_rowing_gait(
    cpg_state: CPGState,
    leading_mask: jnp.ndarray,
    left_mask: jnp.ndarray,
    right_mask: jnp.ndarray,
    left_second_mask: jnp.ndarray,
    right_second_mask: jnp.ndarray,
    max_joint_limit: float,
) -> CPGState:
    """
    Vectorized gait modulation that strictly mirrors the 1-to-1
    coupling logic of the loop-based version.
    """
    num_arms = leading_mask.shape[0]
    all_arms = jnp.arange(num_arms)

    ip_idx, oop_idx = get_oscillator_indices_for_arm(all_arms)

    R = jnp.zeros_like(cpg_state.R)
    X = jnp.zeros_like(cpg_state.X)
    rhos = jnp.zeros_like(cpg_state.rhos)

    # 1. Leading Arms (Only OOP oscillators are biased in X)
    X = X.at[oop_idx].set(jnp.where(leading_mask, max_joint_limit, X[oop_idx]))

    # 2. Intra-arm Coupling (Rower Logic)
    def apply_side_params(mask, bias, curr_R, curr_rhos):
        # Set amplitudes for both oscillators in the arm
        curr_R = curr_R.at[ip_idx].set(jnp.where(mask, max_joint_limit, curr_R[ip_idx]))
        curr_R = curr_R.at[oop_idx].set(jnp.where(mask, max_joint_limit, curr_R[oop_idx]))

        # Set phase bias between IP and OOP oscillators (intra-arm)
        curr_rhos = curr_rhos.at[ip_idx, oop_idx].set(jnp.where(mask, bias, curr_rhos[ip_idx, oop_idx]))
        curr_rhos = curr_rhos.at[oop_idx, ip_idx].set(jnp.where(mask, -bias, curr_rhos[oop_idx, ip_idx]))
        return curr_R, curr_rhos

    # Apply left-side and right-side intra-arm parameters
    R, rhos = apply_side_params(left_mask | left_second_mask, jnp.pi / 2.0, R, rhos)
    R, rhos = apply_side_params(right_mask | right_second_mask, -jnp.pi / 2.0, R, rhos)

    # 3. Secondary Synchronization (Inter-arm 1-to-1 coupling)

    # use fill_value=-1 to handle cases where there are fewer arms than the fixed size
    l_arm_ids = jnp.where(left_second_mask, size=num_arms, fill_value=-1)[0]
    r_arm_ids = jnp.where(right_second_mask, size=num_arms, fill_value=-1)[0]

    l_ip_targets = ip_idx[l_arm_ids]
    r_ip_targets = ip_idx[r_arm_ids]

    valid_pair = (l_arm_ids >= 0) & (r_arm_ids >= 0)

    rhos = rhos.at[l_ip_targets, r_ip_targets].set(jnp.where(valid_pair, jnp.pi, rhos[l_ip_targets, r_ip_targets]))
    rhos = rhos.at[r_ip_targets, l_ip_targets].set(jnp.where(valid_pair, -jnp.pi, rhos[r_ip_targets, l_ip_targets]))

    return cpg_state.replace(R=R, X=X, rhos=rhos)  # type: ignore


def map_cpg_to_brittle_star_actions(outputs: jnp.ndarray, num_arms: int, num_segments: int) -> jnp.ndarray:
    """Broadcasts oscillator outputs to robot joints."""
    cpg_outputs_per_arm = outputs.reshape((num_arms, 2))
    cpg_outputs_per_segment = jnp.repeat(cpg_outputs_per_arm, num_segments, axis=0)
    return cpg_outputs_per_segment.flatten()
