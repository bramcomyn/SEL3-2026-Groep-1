import jax.numpy as jnp

from brittle_star_locomotion.controller.rowing_gait_controller import RowingGaitController
from brittle_star_locomotion.cpg.cpg import CPG, CPGState
from brittle_star_locomotion.cpg.solver import EulerSolver, RK4Solver


def _make_state(num_envs: int = 2, num_arms: int = 5) -> CPGState:
    """Create a deterministic vectorized CPG state for integration-style tests."""
    num_oscillators = num_arms * 2
    amplitude = jnp.full((num_envs, num_oscillators), 0.5)
    zeros = jnp.zeros((num_envs, num_oscillators))
    return CPGState(
        time=jnp.zeros((num_envs,)),
        phase=zeros,
        amplitude=amplitude,
        dot_amplitude=zeros,
        offset=zeros,
        dot_offset=zeros,
        output=zeros,
        target_amplitude=zeros,
        target_offset=zeros,
        intrinsic_frequency=jnp.ones((num_envs, num_oscillators)),
        target_phase_bias=jnp.zeros((num_envs, num_oscillators, num_oscillators)),
    )


def _make_controller_with_real_cpg(state: CPGState, solver) -> RowingGaitController:
    """Build a controller instance that uses the real CPG stepping logic with a chosen solver."""
    cpg = CPG.__new__(CPG)
    cpg.time_step = 0.1
    cpg.solver = solver
    cpg.weights = jnp.zeros((state.phase.shape[0], state.phase.shape[1], state.phase.shape[1]))
    cpg.state = state

    controller = RowingGaitController.__new__(RowingGaitController)
    controller.cpg = cpg  # type: ignore
    return controller


def test_rowing_gait_modulate_and_step_pipeline_updates_vectorized_state_and_actions():
    """End-to-end check: role modulation feeds into CPG stepping and returns mapped actions."""
    initial_state = _make_state(num_envs=2, num_arms=5)
    controller = _make_controller_with_real_cpg(initial_state, EulerSolver())

    # Two environments with different role assignments.
    action = jnp.array(
        [
            [0, 1, 2, 3, 4],
            [0, 2, 1, 4, 3],
        ]
    )

    modulated_state = controller.modulate(initial_state, action, maximal_joint_limit=1.2)
    next_state, actions = controller.step(modulated_state)

    # The modulated state should carry non-zero gait targets while leaving the input unchanged.
    assert jnp.allclose(initial_state.target_amplitude, jnp.zeros_like(initial_state.target_amplitude))
    assert jnp.any(modulated_state.target_amplitude > 0.0)
    assert jnp.any(modulated_state.target_offset > 0.0)

    # Step should advance time, preserve vectorized shape, and map actions from CPG output.
    assert next_state.time.shape == (2,)
    assert next_state.output.shape == (2, 10)
    assert jnp.allclose(next_state.time, jnp.full((2,), 0.1))
    assert jnp.allclose(actions, next_state.output)

    # With non-zero initial amplitude and intrinsic frequency, outputs should become non-zero.
    assert jnp.any(jnp.abs(actions) > 0.0)


def test_rowing_gait_multiple_steps_accumulate_time_and_progress_phase():
    """Sequential stepping should accumulate time and advance oscillator phase consistently."""
    initial_state = _make_state(num_envs=2, num_arms=5)
    controller = _make_controller_with_real_cpg(initial_state, EulerSolver())

    action = jnp.array(
        [
            [0, 1, 2, 3, 4],
            [0, 2, 1, 4, 3],
        ]
    )

    state = controller.modulate(initial_state, action, maximal_joint_limit=1.2)

    for _ in range(3):
        state, actions = controller.step(state)

    # With Euler dt=0.1 and intrinsic frequency=1, phase should advance by 0.3 (weights are zero in this fixture).
    assert jnp.allclose(state.time, jnp.full((2,), 0.3))
    assert jnp.allclose(state.phase, jnp.full((2, 10), 0.3))
    assert jnp.allclose(actions, state.output)


def test_rowing_gait_targets_persist_across_steps_and_change_after_remodulation():
    """Target fields should stay stable during step-only updates and change when modulate is called again."""
    initial_state = _make_state(num_envs=2, num_arms=5)
    controller = _make_controller_with_real_cpg(initial_state, EulerSolver())

    action_a = jnp.array(
        [
            [0, 1, 2, 3, 4],
            [0, 2, 1, 4, 3],
        ]
    )
    action_b = jnp.array(
        [
            [0, 2, 1, 4, 3],
            [0, 1, 2, 3, 4],
        ]
    )

    state_a = controller.modulate(initial_state, action_a, maximal_joint_limit=1.2)
    target_amplitude_a = state_a.target_amplitude
    target_offset_a = state_a.target_offset
    target_phase_bias_a = state_a.target_phase_bias

    stepped_state, _ = controller.step(state_a)
    stepped_state, _ = controller.step(stepped_state)

    # CPG step updates dynamic state but should not rewrite modulation targets.
    assert jnp.allclose(stepped_state.target_amplitude, target_amplitude_a)
    assert jnp.allclose(stepped_state.target_offset, target_offset_a)
    assert jnp.allclose(stepped_state.target_phase_bias, target_phase_bias_a)

    # A new modulation should produce a new target pattern.
    state_b = controller.modulate(stepped_state, action_b, maximal_joint_limit=1.2)
    assert not jnp.allclose(state_b.target_phase_bias, target_phase_bias_a)


def test_rowing_gait_modulate_and_step_pipeline_with_rk4_solver():
    """End-to-end check with RK4 to ensure modulation and stepping integrate correctly."""
    initial_state = _make_state(num_envs=2, num_arms=5)
    controller = _make_controller_with_real_cpg(initial_state, RK4Solver())

    action = jnp.array(
        [
            [0, 1, 2, 3, 4],
            [0, 2, 1, 4, 3],
        ]
    )

    modulated_state = controller.modulate(initial_state, action, maximal_joint_limit=1.2)
    next_state, actions = controller.step(modulated_state)

    assert next_state.time.shape == (2,)
    assert next_state.output.shape == (2, 10)
    assert jnp.allclose(next_state.time, jnp.full((2,), 0.1))
    assert jnp.allclose(actions, next_state.output)
    assert jnp.any(jnp.abs(actions) > 0.0)


def test_rowing_gait_multiple_steps_with_rk4_progress_time_and_phase():
    """Sequential RK4 steps should accumulate time and advance phase in the deterministic fixture."""
    initial_state = _make_state(num_envs=2, num_arms=5)
    controller = _make_controller_with_real_cpg(initial_state, RK4Solver())

    action = jnp.array(
        [
            [0, 1, 2, 3, 4],
            [0, 2, 1, 4, 3],
        ]
    )

    state = controller.modulate(initial_state, action, maximal_joint_limit=1.2)

    for _ in range(3):
        state, actions = controller.step(state)

    # With zero coupling and unit intrinsic frequency, d_phase/dt=1 and phase grows linearly.
    assert jnp.allclose(state.time, jnp.full((2,), 0.3))
    assert jnp.allclose(state.phase, jnp.full((2, 10), 0.3))
    assert jnp.allclose(actions, state.output)
