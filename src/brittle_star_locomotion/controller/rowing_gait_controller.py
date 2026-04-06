import jax.numpy as jnp

from brittle_star_locomotion.config.configuration import Configuration
from brittle_star_locomotion.cpg.cpg import CPG

class RowingGaitController:
    def __init__(self):
        self.configuration = Configuration().configuration
        self.cpg = CPG()

    def modulate(self, action: jnp.ndarray, maximal_joint_limit: float) -> None:
        """Modulate the gait based on the input actions.
        :param action: A jnp.ndarray containing the modulation parameters for the CPG -- shape (number_of_agents,).
        :param maximal_joint_limit: The maximal joint limit for the brittle star.
        """
        # expand masks for each role (shape: number_of_arms,)
        # 0: Leading, 1: Left Rower, 2: Right Rower, 3: Left Secondary, 4: Right Secondary
        masks = action[jnp.newaxis, jnp.newaxis, :] == jnp.arange(5)[jnp.newaxis, :, jnp.newaxis]
        leading_mask, left_rower_mask, right_rower_mask, left_sec_mask, right_sec_mask = masks

        all_arms = jnp.arange(self.configuration.number_of_arms) # TODO: add "number_of_arms" to configuration
        ip_idx, oop_idx = self._get_oscillator_indices(all_arms)

        target_amplitude = jnp.zeros_like(self.cpg.state.target_amplitude)
        target_offset    = jnp.zeros_like(self.cpg.state.target_offset)
        target_phase_bias = jnp.zeros_like(self.cpg.state.target_phase_bias)

        # --- leading arm modulation ---
        
        # modulate leading arm to point upwards (OOP = max, IP = 0)
        target_offset = target_offset.at[:, oop_idx].set(jnp.where(leading_mask, maximal_joint_limit, target_offset[:, oop_idx]))

        # --- rowing arms offset and amplitude modulation ---

        # set rower amplitudes to max for both IP and OOP to make them actively row
        rower_mask = left_rower_mask | right_rower_mask | left_sec_mask | right_sec_mask
        target_amplitude = target_amplitude.at[:, ip_idx].set(jnp.where(rower_mask, maximal_joint_limit, target_amplitude[:, ip_idx]))
        target_amplitude = target_amplitude.at[:, oop_idx].set(jnp.where(rower_mask, maximal_joint_limit, target_amplitude[:, oop_idx]))

        # --- rowing arms phase bias modulation ---

        # intra-arm bias: IP and OOP of the same arm should be out of phase (e.g., π/2 or -π/2) to create a proper rowing motion
        combined_left_mask = left_rower_mask | left_sec_mask
        combined_right_mask = right_rower_mask | right_sec_mask
        
        target_phase_bias = target_phase_bias.at[:, ip_idx, oop_idx].set(jnp.where(combined_left_mask, jnp.pi / 2, target_phase_bias[:, ip_idx, oop_idx]))
        target_phase_bias = target_phase_bias.at[:, ip_idx, oop_idx].set(jnp.where(combined_right_mask, -jnp.pi / 2, target_phase_bias[:, ip_idx, oop_idx]))
        
        # anti-symmetric bias between IP and OOP for the same arm (ensures proper rowing motion)
        target_phase_bias = target_phase_bias.at[:, oop_idx, ip_idx].set(-target_phase_bias[:, ip_idx, oop_idx])

        # inter-arm bias: secondary rowers on opposite sides should be out of phase (e.g., π or -π) to create an alternating rowing pattern
        number_of_oscillators = self.configuration.number_of_oscillators_per_arm * self.configuration.number_of_arms # TODO: add "number_of_oscillators" to configuration
        left_secondary_ip_mask = jnp.zeros(number_of_oscillators, dtype=bool).at[ip_idx].set(left_sec_mask)
        right_secondary_ip_mask = jnp.zeros(number_of_oscillators, dtype=bool).at[ip_idx].set(right_sec_mask)

        inter_arm_pairs = jnp.outer(left_secondary_ip_mask, right_secondary_ip_mask)  # shape: (number_of_oscillators, number_of_oscillators)
        target_phase_bias = target_phase_bias.at[:, :, :].set(jnp.where(inter_arm_pairs, jnp.pi, target_phase_bias))
        target_phase_bias = target_phase_bias.at[:, :, :].set(jnp.where(inter_arm_pairs.transpose(0, 2, 1), -jnp.pi, target_phase_bias))

        # 4. Update CPG state (Ensure these update your CPG object correctly)
        self.cpg.state = self.cpg.state.replace( # type: ignore
            target_amplitude=target_amplitude,
            target_offset=target_offset,
            target_phase_bias=target_phase_bias
        )


    def step(self) -> jnp.ndarray:
        """Update the CPG state and compute the new joint angles.
        :return: A jnp.ndarray containing the joint angles for the brittle star -- shape (number_of_oscillators,).
        """
        self.cpg.step()
        return self._map_cpg_to_brittle_star_actions()
    
    def _get_oscillator_indices(self, arm_index: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get the indices of the oscillators corresponding to a specific arm."""
        return arm_index * 2, arm_index * 2 + 1
        
    def _map_cpg_to_brittle_star_actions(self) -> jnp.ndarray:
        """Map the CPG outputs to the brittle star's joint angles."""
        # Placeholder for mapping logic
        return self.cpg.get_output()
    