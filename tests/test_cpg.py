import jax.numpy as jnp

from brittle_star_locomotion.cpg.cpg import CPG, CPGState


def _make_state(num_envs: int = 2, num_oscillators: int = 4) -> CPGState:
    """Build a deterministic CPGState with all-zero dynamics for unit tests."""
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


def test_cpg_step_returns_updated_state_with_vectorized_shapes():
    """Ensure step computes all updated tensors and keeps vectorized (env, oscillator) shape."""
    cpg = CPG.__new__(CPG)
    cpg.time_step = 0.1
    cpg.state = _make_state(num_envs=2, num_oscillators=4)

    def fake_solver(_current_time, y, _derivative_fn, _delta_time):
        # Returning y + 1 isolates step wiring from numerical solver implementation details.
        return y + 1.0

    cpg.solver = fake_solver # type: ignore

    new_state = CPG.step(cpg, cpg.state)

    expected_output = jnp.ones((2, 4)) + jnp.cos(jnp.ones((2, 4)))

    assert new_state.time.shape == (2,)
    assert new_state.phase.shape == (2, 4)
    assert new_state.amplitude.shape == (2, 4)
    assert new_state.offset.shape == (2, 4)
    assert jnp.allclose(new_state.time, jnp.array([0.1, 0.1]))
    assert jnp.allclose(new_state.phase, jnp.ones((2, 4)))
    assert jnp.allclose(new_state.amplitude, jnp.ones((2, 4)))
    assert jnp.allclose(new_state.offset, jnp.ones((2, 4)))
    assert jnp.allclose(new_state.output, expected_output)


def test_cpg_step_does_not_mutate_state_in_place():
    """Document current API: step returns a new state object instead of mutating self.state."""
    cpg = CPG.__new__(CPG)
    cpg.time_step = 0.1
    cpg.state = _make_state(num_envs=1, num_oscillators=2)

    def fake_solver(_current_time, y, _derivative_fn, _delta_time):
        return y + 1.0

    cpg.solver = fake_solver # type: ignore

    returned_state = CPG.step(cpg)

    assert returned_state is not cpg.state
    assert jnp.allclose(cpg.state.time, jnp.array([0.0]))


def test_cpg_step_can_optionally_update_internal_state():
    """When requested, functional step should also update the instance's internal state."""
    cpg = CPG.__new__(CPG)
    cpg.time_step = 0.1
    cpg.state = _make_state(num_envs=1, num_oscillators=2)

    def fake_solver(_current_time, y, _derivative_fn, _delta_time):
        return y + 1.0

    cpg.solver = fake_solver # type: ignore

    returned_state = CPG.step(cpg, cpg.state, update_internal_state=True)

    assert returned_state is cpg.state
    assert jnp.allclose(cpg.state.time, jnp.array([0.1]))


def test_cpg_get_output_returns_state_output():
    """get_output should act as a simple accessor for the current CPG output tensor."""
    cpg = CPG.__new__(CPG)
    expected = jnp.array([[0.1, -0.2, 0.3]])
    cpg.state = _make_state(num_envs=1, num_oscillators=3).replace(output=expected) # type: ignore

    assert jnp.allclose(CPG.get_output(cpg), expected)
