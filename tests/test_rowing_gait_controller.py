import jax.numpy as jnp

from brittle_star_locomotion.controller.rowing_gait_controller import RowingGaitModulator
from brittle_star_locomotion.cpg.cpg import CPGState


class _FakeCPG:
    def __init__(self, output: jnp.ndarray):
        self._output = output
        self.step_calls = 0
        self.state = _make_state(num_envs=output.shape[0], num_oscillators=output.shape[1]).replace(output=output) # type: ignore

    def step(self, state, *, update_internal_state=False):
        self.step_calls += 1
        # Simulate CPG progression by changing output after each call.
        self._output = self._output + 1.0
        next_state = state.replace(output=self._output)
        if update_internal_state:
            self.state = next_state
        return next_state

    def get_output(self, state=None):
        if state is None:
            return self._output
        return state.output


def _make_state(num_envs: int = 1, num_oscillators: int = 10) -> CPGState:
    zeros = jnp.zeros((num_envs, num_oscillators))
    return CPGState(
        time=jnp.zeros((num_envs,)),
        phase=zeros,
        amplitude=zeros,
        dot_amplitude=zeros,
        offset=zeros,
        dot_offset=zeros,
        output=zeros,
        target_amplitude=zeros,
        target_offset=zeros,
        intrinsic_frequency=zeros,
        target_phase_bias=jnp.zeros((num_envs, num_oscillators, num_oscillators)),
    )


class _Config:
    number_of_arms = 5
    number_of_oscillators_per_arm = 2



def test_get_oscillator_indices_maps_arms_to_ip_oop_pairs():
    """Arm i should map to in-plane index 2i and out-of-plane index 2i+1."""
    controller = RowingGaitModulator.__new__(RowingGaitModulator)

    ip_idx, oop_idx = controller._get_oscillator_indices(jnp.array([0, 1, 4]))

    assert jnp.array_equal(ip_idx, jnp.array([0, 2, 8]))
    assert jnp.array_equal(oop_idx, jnp.array([1, 3, 9]))



def test_map_cpg_to_brittle_star_actions_returns_cpg_output():
    """Current mapping function is identity, so controller output should equal CPG output."""
    controller = RowingGaitModulator.__new__(RowingGaitModulator)
    controller.cpg = _FakeCPG(output=jnp.array([[0.2, -0.4, 0.6]])) # type: ignore
    state = controller.cpg.state # type: ignore

    result = controller._map_cpg_to_brittle_star_actions(state)

    assert jnp.allclose(result, jnp.array([[0.2, -0.4, 0.6]]))



def test_step_functional_returns_new_state_and_actions():
    """Functional step should return both next CPG state and mapped actions."""
    controller = RowingGaitModulator.__new__(RowingGaitModulator)
    controller.cpg = _FakeCPG(output=jnp.array([[1.0, 2.0]])) # type: ignore
    input_state = controller.cpg.state # type: ignore

    next_state, result = controller.step(input_state)

    assert controller.cpg.step_calls == 1 # type: ignore
    assert jnp.allclose(next_state.output, jnp.array([[2.0, 3.0]]))
    assert jnp.allclose(result, jnp.array([[2.0, 3.0]]))


def test_modulate_functional_updates_targets_without_mutating_input_state():
    """Functional modulate should return updated targets and keep input state unchanged."""
    controller = RowingGaitModulator.__new__(RowingGaitModulator)
    controller.configuration = _Config() # type: ignore
    controller.cpg = _FakeCPG(output=jnp.zeros((1, 10))) # type: ignore

    input_state = _make_state(num_envs=1, num_oscillators=10)
    # Roles by arm index: [leading, left-rower, right-rower, left-secondary, right-secondary]
    action = jnp.array([0, 1, 2, 3, 4])

    next_state = controller.modulate(input_state, action, maximal_joint_limit=1.2)

    ip_idx, oop_idx = controller._get_oscillator_indices(jnp.arange(5))

    assert jnp.allclose(input_state.target_amplitude, jnp.zeros_like(input_state.target_amplitude))
    assert jnp.allclose(next_state.target_offset[:, oop_idx[0]], jnp.array([1.2]))
    assert jnp.allclose(next_state.target_amplitude[:, ip_idx[1:]], jnp.full((1, 4), 1.2))
    assert jnp.allclose(next_state.target_amplitude[:, oop_idx[1:]], jnp.full((1, 4), 1.2))
