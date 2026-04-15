import math

import jax.numpy as jnp
import pytest

from brittle_star_locomotion.cpg.solver import EulerSolver, RK4Solver


def test_euler_solver_integrates_constant_derivative():
    """Verify Euler performs one explicit step: y(t+dt) = y(t) + dt * c for constant slope c."""
    solver = EulerSolver()

    y0 = jnp.array([1.0, -2.0])

    def derivative(_t, _y):
        return jnp.array([3.0, 3.0])

    result = solver(current_time=jnp.array(0.0), y=y0, derivative_fn=derivative, delta_time=0.5)

    # Expected: [1, -2] + 0.5 * [3, 3] = [2.5, -0.5]
    assert jnp.allclose(result, jnp.array([2.5, -0.5]))


def test_rk4_solver_matches_exponential_growth_step():
    """Check RK4 accuracy on dy/dt = y, where the exact one-step solution is e^dt."""
    solver = RK4Solver()

    y0 = jnp.array(1.0)

    def derivative(_t, y):
        return y

    dt = 0.1
    result = solver(current_time=jnp.array(0.0), y=y0, derivative_fn=derivative, delta_time=dt)

    # RK4 should be very close to the analytical value for this smooth ODE.
    assert float(result) == pytest.approx(math.exp(dt), rel=1e-4)


def test_euler_solver_handles_vectorized_state():
    """Verify Euler integrates element-wise for vectorized (env, oscillator) states."""
    solver = EulerSolver()

    y0 = jnp.array(
        [
            [1.0, -2.0, 0.0],
            [4.0, 1.0, -3.0],
        ]
    )

    def derivative(_t, _y):
        return jnp.array(
            [
                [0.5, 1.0, -2.0],
                [-1.0, 2.0, 0.25],
            ]
        )

    dt = 0.2
    result = solver(current_time=jnp.array(0.0), y=y0, derivative_fn=derivative, delta_time=dt)

    expected = y0 + dt * derivative(jnp.array(0.0), y0)
    assert result.shape == y0.shape
    assert jnp.allclose(result, expected)


def test_rk4_solver_handles_vectorized_exponential_growth():
    """Check vectorized RK4 on dy/dt = y against analytical y(t+dt) = y(t) * e^dt."""
    solver = RK4Solver()

    y0 = jnp.array(
        [
            [1.0, 2.0, -0.5],
            [3.0, -1.5, 0.25],
        ]
    )

    def derivative(_t, y):
        return y

    dt = 0.1
    result = solver(current_time=jnp.array(0.0), y=y0, derivative_fn=derivative, delta_time=dt)

    expected = y0 * math.exp(dt)
    assert result.shape == y0.shape
    assert jnp.allclose(result, expected, rtol=1e-4, atol=1e-6)
