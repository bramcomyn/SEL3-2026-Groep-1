from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from brittle_star_locomotion.cpg.cpg import CPG, create_cpg_structure
from brittle_star_locomotion.cpg.solver import RK4Solver
from brittle_star_locomotion.gait.gait import map_cpg_to_brittle_star_actions, modulate_rowing_gait

NUM_ARMS: int = 5
NUM_SEGMENTS: int = 3
RENDER_EVERY: int = 5
rng = jax.random.PRNGKey(seed=0)


class Control(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, actions: jnp.ndarray, **kwargs) -> jnp.ndarray:
        pass

    @abstractmethod
    def init(self, **kwargs):
        pass


class CPGControl(Control):
    def __init__(self, control_timestep: float):
        self.control_timestep = control_timestep
        self.num_osc = NUM_ARMS * 2
        self.weights = create_cpg_structure(self.num_osc)

        self.cpg = CPG(weights=self.weights, dt=self.control_timestep, solver=RK4Solver())

        self.jit_step = None

    def init(self, **kwargs):
        """Sets up the modulated rowing gait state."""
        env = kwargs.get("env")
        self.cpg_state = self.cpg.reset(rng=jax.random.PRNGKey(0))

        # Set base frequency
        self.cpg_state = self.cpg_state.replace(omegas=jnp.pi / 2 * jnp.ones_like(self.cpg_state.omegas))  # type: ignore

    def __call__(self, actions: jnp.ndarray, **kwargs) -> jnp.ndarray:
        leading, left, right, left_second, right_second = [0], [1, 2], [3, 4], [2], [4]

        # action_lists = [
        #     [leading],              # Action 0
        #     [left],                 # Action 1
        #     [right],                # Action 2
        #     [left, left_second],    # Action 3
        #     [right, right_second]   # Action 4
        # ]

        # for arm, action in enumerate(actions):
        #     for target in action_lists[action]:
        #         target.append(arm)

        self.cpg_state = modulate_rowing_gait(
            self.cpg_state, leading, left, right, left_second, right_second, max_joint_limit=kwargs.get("max_joint_limit")
        )

        next_cpg_state = self.cpg.step(self.cpg_state)
        self.cpg_state = next_cpg_state
        return map_cpg_to_brittle_star_actions(self.cpg_state.outputs, NUM_ARMS, NUM_SEGMENTS)
