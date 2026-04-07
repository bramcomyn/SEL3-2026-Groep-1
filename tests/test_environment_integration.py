from types import SimpleNamespace

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
        env.reset()

        fixed_roles = jnp.array([0, 1, 2, 3, 4])
        start_position = env.env_state.observations["disk_position"][:, :2]  # type: ignore[index]

        for _ in range(4):
            env_state, reward, terminated, truncated, trajectory = env.step(fixed_roles, num_substeps=50)

        end_position = env_state.observations["disk_position"][:, :2]  # type: ignore[index]
        displacement = jnp.linalg.norm(end_position - start_position, axis=1)

        assert trajectory[1].shape[0] == 50
        assert trajectory[1].shape[-1] == 10
        assert reward.shape == (1,)
        assert terminated.shape == (1,)
        assert truncated.shape == (1,)
        assert jnp.any(jnp.abs(env.cpg_state.output) > 0.0)
        assert jnp.any(displacement > 1e-4)
    finally:
        if original is None:
            Singleton._instances.pop(Configuration, None)
        else:
            Singleton._instances[Configuration] = original
