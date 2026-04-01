from functools import partial

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jax import jit

from brittle_star_locomotion.cpg.equations import CPGEquations
from brittle_star_locomotion.cpg.solver import Solver


@struct.dataclass
class CPGState:
    """
    State container for the Central Pattern Generator.

    :ivar time: Current simulation time.
    :vartype time: float
    :ivar phases: Current phase of each oscillator.
    :vartype phases: jnp.ndarray
    :ivar amplitudes: Current actual amplitude.
    :vartype amplitudes: jnp.ndarray
    :ivar dot_amplitudes: Rate of change for amplitude convergence.
    :vartype dot_amplitudes: jnp.ndarray
    :ivar offsets: Current actual offset (baseline shift).
    :vartype offsets: jnp.ndarray
    :ivar dot_offsets: Rate of change for offset convergence.
    :vartype dot_offsets: jnp.ndarray
    :ivar outputs: The final generated signal (theta) used for control.
    :vartype outputs: jnp.ndarray
    :ivar R: Target amplitudes (set points).
    :vartype R: jnp.ndarray
    :ivar X: Target offsets (set points).
    :vartype X: jnp.ndarray
    :ivar omegas: Intrinsic frequencies.
    :vartype omegas: jnp.ndarray
    :ivar rhos: Target phase biases (coupling matrix).
    :vartype rhos: jnp.ndarray
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
        base_frequency = 2.0 * jnp.pi * 1.0

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
            omegas=jnp.full_like(num_oscillators, base_frequency),
            rhos=jnp.zeros_like(self.weights),
        )


def create_cpg_structure(num_osc: int) -> jnp.ndarray:
    """
    Constructs a symmetric coupling weight matrix for a Central Pattern Generator network.
    
    The network structure couples oscillators in pairs (In-Phase and Out-of-Phase) 
    and links them in a ring topology. The resulting matrix is scaled to ensure
    strong entrainment between units.

    :param num_osc: total number of oscillators in the network (should be even).
    :return: a square weight matrix of shape (num_osc, num_osc).
    """
    weights = jnp.zeros((num_osc, num_osc))

    # identify indices for in-phase (even) and out-of-phase (odd) units
    # note: if num_osc is odd, these arrays will have different lengths, causing a crash
    ip_idx = jnp.arange(0, num_osc, 2)
    oop_idx = jnp.arange(1, num_osc, 2)

    # determine the minimum common length to prevent broadcasting errors
    min_len = jnp.minimum(len(ip_idx), len(oop_idx))
    ip_idx = ip_idx[:min_len]
    oop_idx = oop_idx[:min_len]

    # couple ip and oop oscillators within the same segment
    # this creates the primary functional unit of the cpg
    weights = weights.at[ip_idx, oop_idx].set(1.0)

    # establish a ring topology by coupling each ip unit to the next ip unit
    # jnp.roll ensures the last unit wraps back to the first
    next_ip = jnp.roll(ip_idx, shift=-1)
    weights = weights.at[ip_idx, next_ip].set(1.0)

    # repeat the ring topology for the oop units
    next_oop = jnp.roll(oop_idx, shift=-1)
    weights = weights.at[oop_idx, next_oop].set(1.0)

    # enforce symmetry and apply a global coupling strength multiplier
    # symmetry ensures that if oscillator a affects b, b affects a equally
    weights = 5.0 * jnp.maximum(weights, weights.T)
    
    return weights
