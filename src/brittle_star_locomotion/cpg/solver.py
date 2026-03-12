from abc import ABC, abstractmethod
from typing import Callable

import jax.numpy as jnp


class Solver(ABC):
    """Abstract base class for numerical Ordinary Differential Equation (ODE) solvers."""

    @abstractmethod
    def __call__(self, current_time: float, y: jnp.ndarray, derivative_fn: Callable, delta_time: float) -> jnp.ndarray:
        """Standard interface for numerical ODE solvers.

        :param current_time: The current simulation time (t).
        :param y: The current state value(s) to be integrated.
        :param derivative_fn: A function f(t, y) that returns dy/dt.
        :param delta_time: The time step (dt) for integration.
        :return: The integrated state value(s) at time t + dt.
        """
        pass


class RK4Solver(Solver):
    """Fourth-order Runge-Kutta (RK4) ODE solver.

    Provides high accuracy by sampling the derivative at four different
    points within the time step to compute a weighted average slope.
    """

    def __call__(self, current_time: float, y: jnp.ndarray, derivative_fn: Callable, delta_time: float) -> jnp.ndarray:
        slope1 = derivative_fn(current_time, y)
        slope2 = derivative_fn(current_time + delta_time / 2, y + slope1 * delta_time / 2)
        slope3 = derivative_fn(current_time + delta_time / 2, y + slope2 * delta_time / 2)
        slope4 = derivative_fn(current_time + delta_time, y + slope3 * delta_time)

        average_slope = (slope1 + 2 * slope2 + 2 * slope3 + slope4) / 6
        return y + average_slope * delta_time


class EulerSolver(Solver):
    """First-order Euler ODE solver.

    The simplest integration method, approximating the next state by
    following the tangent line at the current state.
    """

    def __call__(self, current_time: float, y: jnp.ndarray, derivative_fn: Callable, delta_time: float) -> jnp.ndarray:
        slope = derivative_fn(current_time, y)
        return y + delta_time * slope
