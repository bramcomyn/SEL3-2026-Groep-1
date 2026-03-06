from functools import partial

import jax
import jax.numpy as jnp
from flax import struct
from jax import jit, vmap

import chex

from brittle_star_locomotion.cpg.solver import Solver


class CPGEquations:
    @staticmethod
    def phase_de(weights, amplitudes, phases, phase_biases, omegas):
        """Controls the timing.
        Change of the phase over time is defined by the frequency omega_i
        and the influence of other connected oscillators j via coupling weights w_ij and phase biases phi_ij.
        """

        @vmap
        def sine_term(phase_i, phase_biases_i):
            return jnp.sin(phases - phase_i - phase_biases_i)

        couplings = jnp.sum(weights * amplitudes * sine_term(phases, phase_biases), axis=1)
        return omegas + couplings

    @staticmethod
    def critically_dampened_harmonic_oscillator_de(gain, modulator, values, dot_values):
        """Used for amplitude and offset."""
        return gain * ((gain / 4) * (modulator - values) - dot_values)

    @staticmethod
    def first_order_de(_, dot_values):
        return dot_values

    @staticmethod
    def second_order_de(modulator, values, dot_values):
        return CPGEquations.critically_dampened_harmonic_oscillator_de(
            20.0,
            modulator,
            values,
            dot_values,
        )


@struct.dataclass
class CPGState:
    time: float
    phases: jnp.ndarray
    amplitudes: jnp.ndarray
    dot_amplitudes: jnp.ndarray  # rate of change for amplitude convergence
    offsets: jnp.ndarray
    dot_offsets: jnp.ndarray  # rate of change for offset convergence
    outputs: jnp.ndarray  # the final theta values

    R: jnp.ndarray  # target amplitudes
    X: jnp.ndarray  # target offsets
    omegas: jnp.ndarray  # frequencies
    rhos: jnp.ndarray  # phase biases (coupling targets)


class CPG:
    def __init__(self, weights, solver: Solver, dt: float = 0.01):
        self.weights = weights
        self.solver = solver
        self.dt = dt

    @partial(jit, static_argnums=(0,))
    def step(self, state: CPGState) -> CPGState:
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
        """Initializes the CPG state with small random phases to break symmetry."""
        phase_rng, amplitude_rng, offsets_rng = jax.random.split(rng, 3)

        num_oscillators = self.weights.shape[0]

        state = CPGState(
            phases=jax.random.uniform(key=phase_rng, shape=(num_oscillators,), dtype=jnp.float32, minval=-0.01, maxval=0.01),
            amplitudes=jnp.zeros(num_oscillators),
            offsets=jnp.zeros(num_oscillators),
            dot_amplitudes=jnp.zeros(num_oscillators),
            dot_offsets=jnp.zeros(num_oscillators),
            outputs=jnp.zeros(num_oscillators),
            time=0.0,
            R=jnp.zeros(num_oscillators),
            X=jnp.zeros(num_oscillators),
            omegas=jnp.zeros(num_oscillators),
            rhos=jnp.zeros_like(self.weights),
        )

        return state


def init_cpg_state(num_oscillators):
    zeros = jnp.zeros(num_oscillators)

    return CPGState(
        time=0.0,
        phases=zeros,
        amplitudes=zeros,
        dot_amplitudes=zeros,
        offsets=zeros,
        dot_offsets=zeros,
        outputs=zeros,
        R=jnp.ones(num_oscillators) * 0.5,
        X=zeros,
        omegas=jnp.ones(num_oscillators) * 5.0,
        rhos=jnp.zeros((num_oscillators, num_oscillators)),
    )
