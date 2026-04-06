from abc import ABC, abstractmethod
from typing import Callable

import jax.numpy as jnp


class Solver(ABC):
    """abstract base class for numerical ordinary differential equation (ode) solvers."""

    @abstractmethod
    def __call__(self, current_time: float, y: jnp.ndarray, derivative_fn: Callable, delta_time: float) -> jnp.ndarray:
        """standard interface for numerical ode solvers.

        :param current_time: the current simulation time (t).
        :param y: the current state value(s) to be integrated.
        :param derivative_fn: a function f(t, y) that returns dy/dt.
        :param delta_time: the time step (dt) for integration.
        :return: the integrated state value(s) at time t + dt.
        """
        pass


class RK4Solver(Solver):
    """fourth-order runge-kutta (rk4) ode solver.

    provides high accuracy by sampling the derivative at four different
    points within the time step to compute a weighted average slope.
    """

    def __call__(self, current_time: float, y: jnp.ndarray, derivative_fn: Callable, delta_time: float) -> jnp.ndarray:
        """integrates the state using the rk4 method.

        :param current_time: current simulation time.
        :param y: current state values.
        :param derivative_fn: function returning the derivative dy/dt.
        :param delta_time: integration time step.
        :return: state values at the next time step.
        """
        half_dt = delta_time / 2.0

        slope1 = derivative_fn(current_time, y)
        slope2 = derivative_fn(current_time + half_dt, y + slope1 * half_dt)
        slope3 = derivative_fn(current_time + half_dt, y + slope2 * half_dt)
        slope4 = derivative_fn(current_time + delta_time, y + slope3 * delta_time)

        # weighted average of the four sampled slopes
        average_slope = (slope1 + 2.0 * slope2 + 2.0 * slope3 + slope4) / 6.0
        return y + average_slope * delta_time


class EulerSolver(Solver):
    """first-order euler ode solver.

    the simplest integration method, approximating the next state by
    following the tangent line at the current state.
    """

    def __call__(self, current_time: float, y: jnp.ndarray, derivative_fn: Callable, delta_time: float) -> jnp.ndarray:
        """integrates the state using the euler method.

        :param current_time: current simulation time.
        :param y: current state values.
        :param derivative_fn: function returning the derivative dy/dt.
        :param delta_time: integration time step.
        :return: state values at the next time step.
        """
        slope = derivative_fn(current_time, y)
        return y + delta_time * slope
