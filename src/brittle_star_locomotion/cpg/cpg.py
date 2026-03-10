from functools import partial

import chex
import jax
import jax.numpy as jnp
from brittle_star_locomotion.cpg.equations import CPGEquations
from brittle_star_locomotion.cpg.solver import Solver
from flax import struct
from jax import jit


@struct.dataclass
class CPGState:
    """State container for the Central Pattern Generator.

    Attributes:
        time: Current simulation time.
        phases: Current phase of each oscillator.
        amplitudes: Current actual amplitude.
        dot_amplitudes: Rate of change for amplitude convergence.
        offsets: Current actual offset (baseline shift).
        dot_offsets: Rate of change for offset convergence.
        outputs: The final generated signal (theta) used for control.
        R: Target amplitudes (set points).
        X: Target offsets (set points).
        omegas: Intrinsic frequencies.
        rhos: Target phase biases (coupling matrix).
    """

    time: float
    phases: jnp.ndarray
    amplitudes: jnp.ndarray
    dot_amplitudes: jnp.ndarray
    offsets: jnp.ndarray
    dot_offsets: jnp.ndarray
    outputs: jnp.ndarray

    R: jnp.ndarray
    X: jnp.ndarray
    omegas: jnp.ndarray
    rhos: jnp.ndarray


class CPG:
    def __init__(self, weights: jnp.ndarray, solver: Solver, dt: float = 0.01):
        """Initializes the CPG controller.

        :param weights: Adjacency matrix defining oscillator connections.
        :param solver: ODE solver instance (e.g. Euler or RK4).
        :param dt: Time step for integration.
        """
        self.weights = weights
        self.solver = solver
        self.dt = dt

    @partial(jit, static_argnums=(0,))
    def step(self, state: CPGState) -> CPGState:
        """Updates the CPG state by one time step.

        :param state: Current CPGState.
        :return: Updated CPGState.
        """
        new_phases = self.solver(
            state.time, state.phases, lambda t, p: CPGEquations.phase_de(self.weights, state.amplitudes, p, state.rhos, state.omegas), self.dt
        )

        new_dot_amplitudes = self.solver(
            state.time, state.dot_amplitudes, lambda t, da: CPGEquations.second_order_de(state.R, state.amplitudes, da), self.dt
        )

        new_amplitudes = self.solver(state.time, state.amplitudes, lambda t, a: state.dot_amplitudes, self.dt)

        new_dot_offsets = self.solver(state.time, state.dot_offsets, lambda t, do: CPGEquations.second_order_de(state.X, state.offsets, do), self.dt)
        new_offsets = self.solver(state.time, state.offsets, lambda t, o: state.dot_offsets, self.dt)

        new_outputs = new_offsets + new_amplitudes * jnp.cos(new_phases)

        return state.replace(  # type: ignore
            time=state.time + self.dt,
            phases=new_phases,
            amplitudes=new_amplitudes,
            dot_amplitudes=new_dot_amplitudes,
            offsets=new_offsets,
            dot_offsets=new_dot_offsets,
            outputs=new_outputs,
        )

    def reset(self, rng: chex.PRNGKey) -> CPGState:
        """Initializes the CPG state with small random phases to break symmetry.

        :param rng: JAX random key.
        :return: An initialized CPGState.
        """
        num_oscillators = self.weights.shape[0]
        phase_rng = jax.random.split(rng, 1)[0]

        return CPGState(
            time=0.0,
            phases=jax.random.uniform(key=phase_rng, shape=(num_oscillators,), minval=-0.01, maxval=0.01),
            amplitudes=jnp.zeros(num_oscillators),
            dot_amplitudes=jnp.zeros(num_oscillators),
            offsets=jnp.zeros(num_oscillators),
            dot_offsets=jnp.zeros(num_oscillators),
            outputs=jnp.zeros(num_oscillators),
            R=jnp.zeros(num_oscillators),
            X=jnp.zeros(num_oscillators),
            omegas=jnp.zeros(num_oscillators),
            rhos=jnp.zeros_like(self.weights),
        )


def create_cpg_structure(num_osc: int) -> jnp.ndarray:
    """Creates the coupling weight matrix for the CPG network."""
    weights = jnp.zeros((num_osc, num_osc))
    ip_oscillators = jnp.arange(0, num_osc, 2)
    oop_oscillators = jnp.arange(1, num_osc, 2)

    weights = weights.at[ip_oscillators, oop_oscillators].set(1.0)
    next_ip = jnp.roll(ip_oscillators, shift=-1)
    weights = weights.at[ip_oscillators, next_ip].set(1.0)
    next_oop = jnp.roll(oop_oscillators, shift=-1)
    weights = weights.at[oop_oscillators, next_oop].set(1.0)

    weights = 5.0 * jnp.maximum(weights, weights.T)
    return weights
