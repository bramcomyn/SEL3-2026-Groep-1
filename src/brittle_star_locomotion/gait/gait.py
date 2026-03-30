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
    Configures the CPG state by mapping limb-specific roles to synchronized phase-amplitude 
    relationships, enabling coordinated bilateral rowing movements through vectorized coupling.

    :param cpg_state: The current state of the Central Pattern Generator oscillators.
    :param leading_mask: jnp.ndarray (num_arms,) identifying primary leading arms.
    :param left_mask: jnp.ndarray (num_arms,) identifying arms on the left side.
    :param right_mask: jnp.ndarray (num_arms,) identifying arms on the right side.
    :param left_second_mask: jnp.ndarray (num_arms,) identifying secondary/trailing left arms.
    :param right_second_mask: jnp.ndarray (num_arms,) identifying secondary/trailing right arms.
    :param max_joint_limit: Scalar defining the maximum joint excursion or amplitude.
    :return: The updated CPGState after applying phase and amplitude modulations.
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


def map_cpg_to_brittle_star_actions(
    outputs: jnp.ndarray, 
    num_arms: int, 
    num_segments: int
) -> jnp.ndarray:
    """
    Maps centralized oscillator outputs to the distributed joint actuators of the robot.
    
    This function assumes a bipartite control strategy where each arm is driven by 
    a pair of oscillators (e.g., one for lateral and one for longitudinal movement), 
    which are then mirrored across all segments of that specific arm.

    :param outputs: raw CPG state array of shape (num_arms * 2,).
    :param num_arms: total number of arms in the brittle star morphology.
    :param num_segments: number of physical segments (and thus joint pairs) per arm.
    :return: flattened array of joint commands suitable for the MJX environment.
    """
    # group the flat oscillator outputs by arm
    # assumes the cpg was initialized with 2 oscillators per arm
    cpg_outputs_per_arm = outputs.reshape((num_arms, 2))

    # broadcast the arm-level control signals to every segment in the arm
    # this ensures all segments in an arm move in unison (rowing motion)
    # result shape: (num_arms * num_segments, 2)
    cpg_outputs_per_segment = jnp.repeat(cpg_outputs_per_arm, num_segments, axis=0)

    # flatten to a 1d array to match the environment's action space input
    return cpg_outputs_per_segment.flatten()
