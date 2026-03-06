from abc import ABC, abstractmethod

import jax.numpy as jnp


class Solver(ABC):
    @abstractmethod
    def __call__(self, current_time, y, derivative_fn, delta_time) -> jnp.ndarray:
        """Standard interface for numerical ODE solvers."""


class RK4Solver(Solver):
    def __call__(self, current_time, y, derivative_fn, delta_time) -> jnp.ndarray:
        slope1 = derivative_fn(current_time, y)
        slope2 = derivative_fn(current_time + delta_time / 2, y + slope1 * delta_time / 2)
        slope3 = derivative_fn(current_time + delta_time / 2, y + slope2 * delta_time / 2)
        slope4 = derivative_fn(current_time + delta_time, y + slope3 * delta_time)
        average_slope = (slope1 + 2 * slope2 + 2 * slope3 + slope4) / 6
        return y + average_slope * delta_time


class EulerSolver(Solver):
    def __call__(self, current_time, y, derivative_fn, delta_time) -> jnp.ndarray:
        slope = derivative_fn(current_time, y)
        return y + delta_time * slope
