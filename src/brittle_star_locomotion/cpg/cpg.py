import jax.numpy as jnp

from flax import struct

@struct.dataclass
class CPGState:
    """State of a CPG.
    
    :ivar time:                current simulation time                            
    :ivar phase:               current phase of each oscillator                   -- $\\phi_i$
    :ivar amplitude:           current amplitude of each oscillator               -- $r_i$
    :ivar dot_amplitude:       current derivative of amplitude of each oscillator -- $\\dot{r}_i$
    :ivar offset:              current offset of each oscillator                  -- $x_i$
    :ivar dot_offset:          current derivative of offset of each oscillator    -- $\\dot{x}_i$
    :ivar output:              current output of each oscillator                  -- $theta_i = x_i + r_i * \\cos(\\phi_i)$
    :ivar target_amplitude:    target amplitude of each oscillator                -- $R_i$
    :ivar target_offset:       target offset of each oscillator                   -- $X_i$
    :ivar intrinsic_frequency: intrinsic frequency of each oscillator             -- $\\omega_i$
    :ivar target_phase_bias:   target phase bias of each oscillator               -- $\\psi_ij$
    """
    time: float
    phase: jnp.ndarray
    amplitude: jnp.ndarray
    dot_amplitude: jnp.ndarray
    offset: jnp.ndarray
    dot_offset: jnp.ndarray
    output: jnp.ndarray
    target_amplitude: jnp.ndarray
    target_offset: jnp.ndarray
    intrinsic_frequency: jnp.ndarray
    target_phase_bias: jnp.ndarray


class CPG:
    def __init__(self):
        pass

    def step(self, dt):
        pass

    def reset(self, rng):
        pass

    def get_output(self):
        pass
