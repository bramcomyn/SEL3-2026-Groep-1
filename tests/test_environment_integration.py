from types import SimpleNamespace

import jax
import jax.numpy as jnp

from brittle_star_locomotion.config.configuration import Configuration
from brittle_star_locomotion.environment.environment import Environment
from brittle_star_locomotion.util.singleton import Singleton


def _make_test_configuration() -> SimpleNamespace:
    """Build a complete in-memory config needed by Environment + CPG + gait modulator."""
    arena = SimpleNamespace(
        size_x=20.0,
        size_y=20.0,
        sand_ground_color=False,
        attach_target=True,
        wall_height=0.5,
        wall_thickness=0.2,
    )

    env = SimpleNamespace(
        seed=0,
        num_arms=5,
        num_segments_per_arm=5,
        target_distance=2.0,
        simulation_time=10.0,
        num_physics_steps_per_control_step=5,
        time_scale=1.0,
        camera_ids=[0],
        render_size_x=240,
        render_size_y=240,
        num_substeps_per_modulation=50,
        arena=arena,
    )

    rl = SimpleNamespace(
        number_of_environments=1,
        observations_to_use=None,
    )

    return SimpleNamespace(
        env=env,
        rl=rl,
        cpg_seed=0,
        cpg_solver="rk4",
        number_of_oscillators=10,
        number_of_environments=1,
        cpg_time_step=0.05,
        cpg_base_frequency=1.0,
        cpg_coupling_strength=1.0,
        number_of_arms=5,
        number_of_oscillators_per_arm=2,
    )


class _InjectedConfiguration:
    def __init__(self, config: SimpleNamespace):
        self.configuration = config


def test_environment_step_with_real_biorobot_stack_moves_robot():
    """Real integration test: fixed arm roles should produce non-zero brittle star displacement."""
    original = Singleton._instances.get(Configuration)
    Singleton._instances[Configuration] = _InjectedConfiguration(_make_test_configuration())

    try:
        env = Environment()
        env_state, cpg_state = env.reset()
        fixed_roles = jnp.array([0, 1, 2, 3, 4])
        trajectory_env_states = None
        trajectory_cpg_outputs = None
        for _ in range(4):
            env_state, cpg_state, _, _, _, trajectory = env.step(
                env_state,
                cpg_state,
                fixed_roles,
                num_substeps=50,
            )
            substep_env_states = trajectory[0]
            substep_cpg_outputs = trajectory[1]

            if trajectory_env_states is None:
                trajectory_env_states = substep_env_states
            else:
                trajectory_env_states = jax.tree_util.tree_map(
                    lambda a, b: jnp.concatenate((a, b), axis=0),
                    trajectory_env_states,
                    substep_env_states,
                )

            if trajectory_cpg_outputs is None:
                trajectory_cpg_outputs = substep_cpg_outputs
            else:
                trajectory_cpg_outputs = jnp.concatenate(
                    (trajectory_cpg_outputs, substep_cpg_outputs),
                    axis=0,
                )

        assert trajectory_env_states is not None
        trajectory = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), trajectory_env_states)
        start_position = trajectory.observations["disk_position"][:, 0, :2]  # type: ignore[index]
        end_position = trajectory.observations["disk_position"][:, -1, :2]  # type: ignore[index]
        displacement = jnp.linalg.norm(end_position - start_position, axis=1)

        assert trajectory.observations["disk_position"].shape[0] == 1  # type: ignore[index]
        assert trajectory.observations["disk_position"].shape[1] == 200  # type: ignore[index]
        assert trajectory_cpg_outputs is not None
        assert jnp.any(jnp.abs(trajectory_cpg_outputs) > 0.0)
        assert jnp.any(displacement > 1e-4)
    finally:
        if original is None:
            Singleton._instances.pop(Configuration, None)
        else:
            Singleton._instances[Configuration] = original
