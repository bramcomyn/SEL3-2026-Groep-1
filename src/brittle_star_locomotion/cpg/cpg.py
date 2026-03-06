from functools import partial

import jax.numpy as jnp
from flax import struct
from jax import jit, vmap

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
        # update phase using phase_de
        def phase_de(_, p):
            return CPGEquations.phase_de(self.weights, state.amplitudes, p, state.rhos, state.omegas)

        new_phases = self.solver(state.time, state.phases, phase_de, self.dt)

        # update derivatives for amplitude
        def amplitudes_de(t, da):
            CPGEquations.critically_dampened_harmonic_oscillator_de(20.0, state.R, state.amplitudes, da)

        new_dot_amplitudes = self.solver(state.time, state.dot_amplitudes, amplitudes_de, self.dt)
        new_amplitudes = state.amplitudes + new_dot_amplitudes * self.dt

        # update derivatives for offsets
        def offsets_de(t, do):
            CPGEquations.critically_dampened_harmonic_oscillator_de(20.0, state.X, state.offsets, do)

        new_dot_offsets = self.solver(state.time, state.dot_offsets, offsets_de, self.dt)
        new_offsets = state.offsets + new_dot_offsets * self.dt

        # calculate final output
        new_outputs = state.offsets + new_amplitudes * jnp.cos(new_phases)

        return state.replace(  # type: ignore
            time=state.time + self.dt,
            phases=new_phases,
            amplitudes=new_amplitudes,
            dot_amplitudes=new_dot_amplitudes,
            offsets=new_offsets,
            dot_offsets=new_dot_offsets,
            outputs=new_outputs,
        )


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
