import jax

import jax.numpy as jnp

from flax import struct

from brittle_star_locomotion.config.configuration import Configuration
from brittle_star_locomotion.cpg.solver import Solver, RK4Solver, EulerSolver
from brittle_star_locomotion.cpg.equations import CPGEquations

@struct.dataclass
class CPGState:
    """State of a CPG.

    When initialised, the shapes will be (number_of_environments, number_of_oscillators) to allow for vectorised computation across multiple environments.
    
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
    time:                jnp.ndarray # shape: (number_of_environments,)
    phase:               jnp.ndarray # shape: (number_of_environments, number_of_oscillators)
    amplitude:           jnp.ndarray # shape: (number_of_environments, number_of_oscillators)
    dot_amplitude:       jnp.ndarray # shape: (number_of_environments, number_of_oscillators)
    offset:              jnp.ndarray # shape: (number_of_environments, number_of_oscillators)
    dot_offset:          jnp.ndarray # shape: (number_of_environments, number_of_oscillators)
    output:              jnp.ndarray # shape: (number_of_environments, number_of_oscillators)
    target_amplitude:    jnp.ndarray # shape: (number_of_environments, number_of_oscillators)
    target_offset:       jnp.ndarray # shape: (number_of_environments, number_of_oscillators)
    intrinsic_frequency: jnp.ndarray # shape: (number_of_environments, number_of_oscillators)
    target_phase_bias:   jnp.ndarray # shape: (number_of_environments, number_of_oscillators, number_of_oscillators)


class CPG:
    def __init__(self):
        """Initialise a new CPG"""
        self.configuration  = Configuration().configuration
        self.rng = jax.random.PRNGKey(self.configuration.cpg.seed)
        
        self.solver: Solver = RK4Solver() if self.configuration.cpg.solver == 'rk4' else EulerSolver()
        self.number_of_oscillators = self.configuration.environment.number_of_arms * 2
        self.number_of_environments = self.configuration.environment.number_of_environments
        self.time_step = self.configuration.cpg.time_step

        self._initialise_weights()
        self.state = self.reset()

    def step(self, state: CPGState | None = None, *, update_internal_state: bool = False) -> CPGState:
        """Advance one CPG step in a functional way.

        :param state: input CPG state. If None, self.state is used.
        :param update_internal_state: when True, write the returned state to self.state.
        :return: next CPG state.
        """
        state = self.state if state is None else state

        new_phase = self.solver(state.time, state.phase, lambda t, y: self._phase_dynamics(state, t, y), self.time_step)

        new_dot_amplitude = self.solver(
            state.time,
            state.dot_amplitude,
            lambda t, y: self._amplitude_acceleration(state, t, y),
            self.time_step,
        )

        new_amplitude = self.solver(
            state.time,
            state.amplitude,
            lambda t, y: self._amplitude_velocity(state, t, y),
            self.time_step,
        )

        new_dot_offset = self.solver(
            state.time,
            state.dot_offset,
            lambda t, y: self._offset_acceleration(state, t, y),
            self.time_step,
        )

        new_offset = self.solver(
            state.time,
            state.offset,
            lambda t, y: self._offset_velocity(state, t, y),
            self.time_step,
        )

        new_output = new_offset + new_amplitude * jnp.cos(new_phase)

        next_state = state.replace(  # type: ignore
            time=state.time + self.time_step,
            phase=new_phase,
            amplitude=new_amplitude,
            dot_amplitude=new_dot_amplitude,
            offset=new_offset,
            dot_offset=new_dot_offset,
            output=new_output,
        )

        if update_internal_state:
            self.state = next_state

        return next_state

    def reset(self) -> CPGState:
        """Reset the CPG to its initial state.
        
        This method initializes the state of the CPG, including the phase, amplitude, offset, and other relevant variables for each oscillator.
        The initial phase is set to small random values to break symmetry and encourage diverse oscillation patterns across the oscillators.
        The intrinsic frequency is set to a base value defined in the configuration, and the weights are initialized based on the specified coupling structure.

        :return: The initial state of the CPG after reset.
        """
        self.rng, phase_rng = jax.random.split(self.rng)
        initial_phase = jax.random.uniform(
            phase_rng,
            shape=(self.number_of_environments, self.number_of_oscillators),
            minval=-0.1,
            maxval=0.1,
        )

        base_frequency = self.configuration.cpg.base_frequency_multiplier * jnp.pi
        
        state = CPGState(
            time                = jnp.zeros(self.number_of_environments),
            phase               = initial_phase,
            amplitude           = jnp.zeros((self.number_of_environments, self.number_of_oscillators)),
            dot_amplitude       = jnp.zeros((self.number_of_environments, self.number_of_oscillators)),
            offset              = jnp.zeros((self.number_of_environments, self.number_of_oscillators)),
            dot_offset          = jnp.zeros((self.number_of_environments, self.number_of_oscillators)),
            output              = jnp.zeros((self.number_of_environments, self.number_of_oscillators)),
            target_amplitude    = jnp.zeros((self.number_of_environments, self.number_of_oscillators)),
            target_offset       = jnp.zeros((self.number_of_environments, self.number_of_oscillators)),
            intrinsic_frequency = jnp.full((self.number_of_environments, self.number_of_oscillators), base_frequency),
            target_phase_bias   = jnp.zeros_like(self.weights)
        )

        return state
    

    def get_output(self, state: CPGState | None = None):
        """Get the current output of the CPG, which is calculated as the sum of the offset and the product of amplitude and cosine of phase for each oscillator.
        
        :return: current output of each oscillator, which represents the signal that would be sent to the motors in a robotic implementation. 
            This output is what ultimately drives the locomotion pattern generated by the CPG.
        """
        state = self.state if state is None else state
        return state.output

    def _phase_dynamics(self, state: CPGState, _time: float, phase: jnp.ndarray) -> jnp.ndarray:
        """Calculate the phase dynamics of the CPG based on the current state and weights.
        
        :param _time: current simulation time (not used in this calculation but included for consistency with solver interface)
        :param phase: current phase of each oscillator
        :return: rate of change of phase for each oscillator, which is calculated using a phase coupling function that incorporates the weights, current amplitude, target phase bias, and intrinsic frequency of each oscillator. 
             This function captures how the oscillators influence each other's phase based on their coupling structure and current state, which is essential for generating coordinated oscillations that underlie locomotion patterns.
        """
        return jax.vmap(CPGEquations.phase_de)(
            self.weights,
            state.amplitude,
            phase,
            state.target_phase_bias,
            state.intrinsic_frequency,
        )

    def _amplitude_acceleration(self, state: CPGState, _time: float, dot_amplitude: jnp.ndarray) -> jnp.ndarray:
        """Calculate the amplitude acceleration of the CPG based on the current state and target amplitude.

        :param _time: current simulation time (not used in this calculation but included for consistency with solver interface)
        :param dot_amplitude: current derivative of amplitude of each oscillator (not used in this calculation but included for consistency with solver interface)
        :return: acceleration of amplitude of each oscillator, which is calculated using a second-order differential equation that drives the amplitude towards its target value. 
            This allows the amplitude to change smoothly over time, which is important for generating naturalistic locomotion patterns.
        """
        return CPGEquations.second_order_de(
            state.target_amplitude,
            state.amplitude,
            dot_amplitude
        )

    def _amplitude_velocity(self, state: CPGState, _time: float, _amplitude: jnp.ndarray) -> jnp.ndarray:
        """Calculate the amplitude velocity of the CPG, which is simply the current derivative of amplitude.
        
        :param _time: current simulation time (not used in this calculation but included for consistency with solver interface)
        :param _amplitude: current amplitude of each oscillator (not used in this calculation but included for consistency with solver interface)
        :return: current derivative of amplitude of each oscillator, which represents the velocity of the amplitude change.
            This is used by the solver to update the amplitude in the next time step.
        """
        return state.dot_amplitude

    def _offset_acceleration(self, state: CPGState, _time: float, dot_offset: jnp.ndarray) -> jnp.ndarray:
        """Calculate the offset acceleration of the CPG based on the current state and target offset.
        
        :param _time: current simulation time (not used in this calculation but included for consistency with solver interface)
        :param dot_offset: current derivative of offset of each oscillator (not used in this calculation but included for consistency with solver interface)
        :return: acceleration of offset of each oscillator, which is calculated using a second-order differential equation that drives the offset towards its target value. 
            This allows the offset to change smoothly over time, which is important for generating naturalistic locomotion patterns.
        """
        return CPGEquations.second_order_de(
            state.target_offset,
            state.offset,
            dot_offset
        )

    def _offset_velocity(self, state: CPGState, _time: float, _offset: jnp.ndarray) -> jnp.ndarray:
        """Calculate the offset velocity of the CPG, which is simply the current derivative of offset.
        
        :param _time: current simulation time (not used in this calculation but included for consistency with solver interface)
        :param _offset: current offset of each oscillator (not used in this calculation but included for consistency with solver interface)
        :return: current derivative of offset of each oscillator, which represents the velocity of the offset change.
            This is used by the solver to update the offset in the next time step.
        """
        return state.dot_offset

    def _initialise_weights(self):
        """Initialise the weights of the CPG based on the configuration.

        The weights are structured into a network that couples oscillators in pairs (in-plane and out-of-plane) to achieve the desired locomotion pattern.
        This specific structure links the oscillators in a ring topology.
        The resulting matrix is scaled to ensure strong entrainment between the oscillators, which is crucial for coordinated movement.
        """
        self.weights = jnp.zeros((self.number_of_environments, self.number_of_oscillators, self.number_of_oscillators))

        # identify in-plane and out-of-plane oscillators based on their indices
        ip_idx  = jnp.arange(0, self.number_of_oscillators, 2)
        oop_idx = jnp.arange(1, self.number_of_oscillators, 2)

        # determine minimum common length to prevent broadcasting issues
        min_length = min(len(ip_idx), len(oop_idx))
        ip_idx     = ip_idx[:min_length]
        oop_idx    = oop_idx[:min_length]

        # couple ip and oop oscillators in same segment
        # this creates the primary functional unit of the cpg
        self.weights = self.weights.at[:, ip_idx, oop_idx].set(1.0)

        # establish a ring topology by coupling each ip unit to the next ip unit and each oop unit to the next oop unit
        # jnp.roll ensures the last unit wraps back to the first, creating a closed loop
        next_ip_idx = jnp.roll(ip_idx, shift=-1)
        self.weights = self.weights.at[:, ip_idx, next_ip_idx].set(1.0)

        next_oop_idx = jnp.roll(oop_idx, shift=-1)
        self.weights = self.weights.at[:, oop_idx, next_oop_idx].set(1.0)

        # enforce symmetry and apply global coupling strength multiplier
        self.weights = self.configuration.cpg.coupling_strength * jnp.maximum(self.weights, self.weights.transpose((0, 2, 1)))
