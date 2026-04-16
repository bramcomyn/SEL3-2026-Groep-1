import jax.numpy as jnp

from brittle_star_locomotion.cpg.equations import CPGEquations


def test_phase_de_returns_omegas_when_no_coupling():
    """With zero coupling weights, phase dynamics should reduce to intrinsic frequency only."""
    weights = jnp.zeros((2, 2))
    amplitudes = jnp.ones((2, 2))
    phases = jnp.array([0.2, -0.4])
    phase_biases = jnp.zeros((2, 2))
    omegas = jnp.array([1.3, 2.1])

    result = CPGEquations.phase_de(weights, amplitudes, phases, phase_biases, omegas)

    # No coupling contribution means d_phase/dt equals omega element-wise.
    assert jnp.allclose(result, omegas)


def test_critically_dampened_harmonic_oscillator_de_formula():
    """Ensure helper returns exactly the documented closed-form acceleration formula."""
    gain = 10.0
    modulator = jnp.array([2.0, -1.0])
    values = jnp.array([1.0, 3.0])
    dot_values = jnp.array([0.5, -0.5])

    result = CPGEquations.critically_dampened_harmonic_oscillator_de(gain, modulator, values, dot_values)
    expected = gain * ((gain / 4.0) * (modulator - values) - dot_values)

    assert jnp.allclose(result, expected)


def test_second_order_de_uses_gain_20():
    """Confirm second_order_de is a thin wrapper around the helper with fixed gain = 20."""
    modulator = jnp.array([1.0, 0.0])
    values = jnp.array([0.0, 0.5])
    dot_values = jnp.array([0.2, -0.1])

    expected = CPGEquations.critically_dampened_harmonic_oscillator_de(20.0, modulator, values, dot_values)
    result = CPGEquations.second_order_de(modulator, values, dot_values)

    assert jnp.allclose(result, expected)
