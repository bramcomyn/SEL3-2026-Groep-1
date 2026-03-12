import jax.numpy as jnp
from jax import vmap


class CPGEquations:
    """Mathematical definitions for the Central Pattern Generator dynamics."""

    @staticmethod
    def phase_de(weights: jnp.ndarray, amplitudes: jnp.ndarray, phases: jnp.ndarray, phase_biases: jnp.ndarray, omegas: jnp.ndarray) -> jnp.ndarray:
        """Calculates the derivative of the phase (timing).

        The change of phase over time is defined by the intrinsic frequency (omega)
        and the coupling influence of neighboring oscillators.

        :param weights: Coupling strength matrix (num_oscillators, num_oscillators).
        :param amplitudes: Current oscillator amplitudes.
        :param phases: Current oscillator phases.
        :param phase_biases: Target phase differences between oscillators.
        :param omegas: Intrinsic frequencies.
        :return: d_phase/dt.
        """

        @vmap
        def sine_term(phase_i: jnp.ndarray, phase_biases_i: jnp.ndarray) -> jnp.ndarray:
            return jnp.sin(phases - phase_i - phase_biases_i)

        couplings = jnp.sum(weights * amplitudes * sine_term(phases, phase_biases), axis=1)
        return omegas + couplings

    @staticmethod
    def critically_dampened_harmonic_oscillator_de(gain: float, modulator: jnp.ndarray, values: jnp.ndarray, dot_values: jnp.ndarray) -> jnp.ndarray:
        """Second-order ODE to smoothly converge a value to a target modulator."""
        return gain * ((gain / 4) * (modulator - values) - dot_values)

    @staticmethod
    def second_order_de(modulator: jnp.ndarray, values: jnp.ndarray, dot_values: jnp.ndarray) -> jnp.ndarray:
        """Applies a critically dampened response with a fixed gain of 20.0."""
        return CPGEquations.critically_dampened_harmonic_oscillator_de(
            20.0,
            modulator,
            values,
            dot_values,
        )
