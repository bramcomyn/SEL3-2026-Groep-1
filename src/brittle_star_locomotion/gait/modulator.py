import jax
import jax.numpy as jnp

from brittle_star_locomotion.configuration import Configuration
from brittle_star_locomotion.cpg.cpg import CPG, CPGState, create_cpg_structure
from brittle_star_locomotion.cpg.solver import RK4Solver
from brittle_star_locomotion.gait.gait import modulate_rowing_gait, map_cpg_to_brittle_star_actions

class Modulator:
    """
    Modulator class for controlling the gait of the brittle star.
    This class will be responsible for generating the appropriate signals
    to control the movement of the brittle star based on the desired gait pattern.
    It will take into account the configuration of the environment and the current state
    of the brittle star to produce the necessary outputs for locomotion.
    """
    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.cpg = CPG(
            weights=create_cpg_structure(configuration.get("num_oscillators", 10)),
            solver=RK4Solver(),
            dt=configuration.get("dt", 0.01)
        )

        self.rng = jax.random.PRNGKey(configuration.get("cpg_random_seed", 42))
        self.state: CPGState = self.cpg.reset(rng=self.rng)

    def __update(self, actions: jnp.ndarray, max_joint_limits: float):
        """Updates the internal state of the modulator based on the provided actions.
        This method will process the input actions, update any relevant internal variables,
        and prepare the modulator for generating the next set of gait signals.
        :param actions: A jnp.ndarray containing the control actions or parameters for modulation.
        :param max_joint_limits: The maximum joint limits for normalization.
        """
        self.state = modulate_rowing_gait(
            self.state,
            actions,
            max_joint_limits
        )

    def output(self) -> jnp.ndarray:
        """Generates the output signals for controlling the brittle star's movement.
        This method will take the current state of the modulator and produce the necessary
        output signals that can be used to control the joints of the brittle star.
        :return: A jnp.ndarray containing the generated gait signals to be used for controlling the brittle star's movement.
        """
        return map_cpg_to_brittle_star_actions(
            self.state.outputs,
            self.configuration.get("num_arms", 5),
            self.configuration.get("num_joints_per_arm", 3)
        )

    def modulate(self, actions: jnp.ndarray, max_joint_limits: float):
        """Update the internal state based on the provided actions and generate the corresponding gait signals for control.
        :param actions: A jnp.ndarray containing the control actions or parameters for modulation.
        :param max_joint_limits: The maximum joint limits for normalization.
        """
        self.__update(actions, max_joint_limits)
