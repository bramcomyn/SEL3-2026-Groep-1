import jax.numpy as jnp

from brittle_star_locomotion.config.configuration import Configuration
from brittle_star_locomotion.cpg.cpg import CPG, CPGState


class RowingGaitModulator:
    """Functional gait modulator built on top of the CPG model."""

    def __init__(self):
        self.configuration = Configuration().configuration
        self.cpg = CPG()

    def modulate(self, cpg_state: CPGState, action: jnp.ndarray, maximal_joint_limit: float) -> CPGState:
        """Return a new CPG state with targets modulated from discrete arm roles.

        Roles per arm:
        0: leading, 1: left rower, 2: right rower, 3: left secondary, 4: right secondary.
        """
        if action.ndim not in (1, 2):
            raise ValueError("action must have shape (arms,) or (envs, arms)")

        if action.ndim == 1:
            action = action[jnp.newaxis, :]

        num_envs, num_arms = action.shape
        if cpg_state.target_amplitude.shape[0] != num_envs:
            raise ValueError("action environment dimension must match cpg_state")

        # 0: leading, 1: right rower, 2: left rower, 3: right secondary, 4: left secondary
        masks = action[:, jnp.newaxis, :] == jnp.arange(5)[jnp.newaxis, :, jnp.newaxis]
        leading_mask     = masks[:, 0, :]
        right_rower_mask = masks[:, 1, :]
        left_rower_mask  = masks[:, 2, :]
        right_sec_mask   = masks[:, 3, :]
        left_sec_mask    = masks[:, 4, :]

        all_arms = jnp.arange(num_arms)
        ip_idx, oop_idx = self._get_oscillator_indices(all_arms)

        target_amplitude = jnp.zeros_like(cpg_state.target_amplitude)
        target_offset = jnp.zeros_like(cpg_state.target_offset)
        target_phase_bias = jnp.zeros_like(cpg_state.target_phase_bias)

        # Leading arm points upward by offsetting its OOP oscillator.
        target_offset = target_offset.at[:, oop_idx].set(jnp.where(leading_mask, maximal_joint_limit, target_offset[:, oop_idx]))

        # Rowing roles are active oscillators in both IP and OOP.
        rower_mask = left_rower_mask | right_rower_mask | left_sec_mask | right_sec_mask
        target_amplitude = target_amplitude.at[:, ip_idx].set(jnp.where(rower_mask, maximal_joint_limit, target_amplitude[:, ip_idx]))
        target_amplitude = target_amplitude.at[:, oop_idx].set(jnp.where(rower_mask, maximal_joint_limit, target_amplitude[:, oop_idx]))

        # Intra-arm bias: IP and OOP are anti-symmetric.
        combined_left_mask = left_rower_mask | left_sec_mask
        combined_right_mask = right_rower_mask | right_sec_mask
        intra_arm_bias = jnp.where(combined_left_mask, jnp.pi / 2, jnp.where(combined_right_mask, -jnp.pi / 2, 0.0))
        target_phase_bias = target_phase_bias.at[:, ip_idx, oop_idx].set(intra_arm_bias)
        target_phase_bias = target_phase_bias.at[:, oop_idx, ip_idx].set(-intra_arm_bias)

        # Inter-arm bias: opposite secondary rowers are anti-phase.
        number_of_oscillators = cpg_state.target_amplitude.shape[1]
        left_secondary_ip_mask = jnp.zeros((num_envs, number_of_oscillators), dtype=bool).at[:, ip_idx].set(left_sec_mask)
        right_secondary_ip_mask = jnp.zeros((num_envs, number_of_oscillators), dtype=bool).at[:, ip_idx].set(right_sec_mask)

        inter_arm_pairs = left_secondary_ip_mask[:, :, jnp.newaxis] & right_secondary_ip_mask[:, jnp.newaxis, :]
        target_phase_bias = jnp.where(inter_arm_pairs, jnp.pi, target_phase_bias)
        target_phase_bias = jnp.where(inter_arm_pairs.transpose((0, 2, 1)), -jnp.pi, target_phase_bias)

        return cpg_state.replace(  # type: ignore
            target_amplitude=target_amplitude,
            target_offset=target_offset,
            target_phase_bias=target_phase_bias,
        )

    def step(self, cpg_state: CPGState) -> tuple[CPGState, jnp.ndarray]:
        """Advance CPG one step and map the resulting output to joint actions."""
        next_state = self.cpg.step(cpg_state)
        actions = self._map_cpg_to_brittle_star_actions(next_state)
        return next_state, actions
    
    def _get_oscillator_indices(self, arm_index: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get the indices of the oscillators corresponding to a specific arm."""
        return arm_index * 2, arm_index * 2 + 1
        
    def _map_cpg_to_brittle_star_actions(self, cpg_state: CPGState) -> jnp.ndarray:
        """Map the CPG outputs to the brittle star's joint angles."""
        return self.cpg.get_output(cpg_state)
    