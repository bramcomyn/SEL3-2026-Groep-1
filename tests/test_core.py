import pytest
from brittle_star_locomotion import core

# Short duration to ensure tests run quickly while still triggering JIT
TEST_SIM_TIME = 0.5


@pytest.fixture(scope="module")
def sim_setup():
    """
    Fixture to initialize simulation objects once per test module.
    Returns the environment wrapper, CPG controller, initial state, and step count.
    """
    env, cpg, env_config = core.setup_simulation_objects(simulation_time=TEST_SIM_TIME)
    cpg_state = core.initialize_gait(cpg, env)
    num_steps = int(TEST_SIM_TIME / env_config.control_timestep)
    return env, cpg, cpg_state, num_steps


def get_qpos(trajectory):
    """
    Helper to extract the physics state (qpos) from the MJX trajectory.
    Handles different Brax/MJX state structures by checking for pipeline_state.
    """
    if hasattr(trajectory, "pipeline_state"):
        # Standard Brax/MJX structure
        return trajectory.pipeline_state.q
    # Fallback for raw MJX data wrappers
    qpos = trajectory.mj_data.qpos
    return qpos


def test_initialization(sim_setup):
    """
    Verify that the Environment and CPG are instantiated with correct dimensions.
    Checks that the CPG state matches the expected number of oscillators (2 per arm).
    """
    env, cpg, cpg_state, _ = sim_setup
    assert env is not None, "Environment failed to initialize"
    assert cpg is not None, "CPG failed to initialize"
    assert cpg_state.omegas.shape == (core.NUM_ARMS * 2,), "CPG state dimension mismatch"


@pytest.mark.parametrize("invalid_time", [-1.0, 0.0])
def test_invalid_sim_time(invalid_time):
    """
    Boundary value testing for the simulation setup.
    Ensures that negative or zero time raises a clear ValueError rather than crashing JAX.
    """
    with pytest.raises(ValueError, match="simulation_time must be positive"):
        core.setup_simulation_objects(simulation_time=invalid_time)
