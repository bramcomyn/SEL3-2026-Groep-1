import jax.numpy as jnp


class CPGEquations:
    """mathematical definitions for the central pattern generator dynamics."""

    @staticmethod
    def phase_de(weights: jnp.ndarray, amplitudes: jnp.ndarray, phases: jnp.ndarray, phase_biases: jnp.ndarray, omegas: jnp.ndarray) -> jnp.ndarray:
        """calculates the derivative of the phase (timing).

        the change of phase over time is defined by the intrinsic frequency (omega)
        and the coupling influence of neighboring oscillators.

        :param weights: coupling strength matrix (num_oscillators, num_oscillators).
        :param amplitudes: current oscillator amplitudes.
        :param phases: current oscillator phases.
        :param phase_biases: target phase differences between oscillators.
        :param omegas: intrinsic frequencies.
        :return: d_phase/dt.
        """
        # compute the phase differences between all oscillator pairs (i, j)
        # phases[None, :] - phases[:, None] creates a (n, n) matrix of (phase_j - phase_i)
        phase_diffs = phases[None, :] - phases[:, None]

        # calculate sine terms with the target phase biases
        # phi_j - phi_i - rho_ij
        coupling_terms = jnp.sin(phase_diffs - phase_biases)

        # apply weights and amplitudes to determine final coupling influence
        # sum across axis 1 to get the net effect on each oscillator i
        couplings = jnp.sum(weights * amplitudes * coupling_terms, axis=1)

        return omegas + couplings

    @staticmethod
    def critically_dampened_harmonic_oscillator_de(gain: float, modulator: jnp.ndarray, values: jnp.ndarray, dot_values: jnp.ndarray) -> jnp.ndarray:
        """second-order ode to smoothly converge a value to a target modulator.

        :param gain: proportional gain determining convergence speed.
        :param modulator: the target set point (e.g. R or X).
        :param values: the current state values (e.g. amplitudes or offsets).
        :param dot_values: the current rate of change (velocity).
        :return: the second derivative (acceleration) of the values.
        """
        return gain * ((gain / 4.0) * (modulator - values) - dot_values)

    @staticmethod
    def second_order_de(modulator: jnp.ndarray, values: jnp.ndarray, dot_values: jnp.ndarray) -> jnp.ndarray:
        """applies a critically dampened response with a fixed gain of 20.0.

        :param modulator: the target set point.
        :param values: the current state values.
        :param dot_values: the current rate of change.
        :return: acceleration required to converge to the modulator.
        """
        return CPGEquations.critically_dampened_harmonic_oscillator_de(
            20.0,
            modulator,
            values,
            dot_values,
        )
