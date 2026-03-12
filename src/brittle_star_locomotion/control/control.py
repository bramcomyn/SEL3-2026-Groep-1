# https://stackoverflow.com/questions/5748946/pythonic-way-to-resolve-circular-import-statements
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import brittle_star_locomotion.environment.environment as environment
import jax
import jax.numpy as jnp
from biorobot.brittle_star.environment.directed_locomotion.shared import BaseEnvState
from brittle_star_locomotion.cpg.cpg import CPG, create_cpg_structure
from brittle_star_locomotion.cpg.solver import RK4Solver
from brittle_star_locomotion.gait.gait import modulate_rowing_gait

NUM_ARMS: int = 5
NUM_SEGMENTS: int = 3
RENDER_EVERY: int = 5


class Control:
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, env: environment.Environment, actions: jnp.ndarray) -> BaseEnvState:
        pass


class CPGControl(Control):
    def __init__(self, control_timestep: float):
        self.num_osc = NUM_ARMS * 2
        self.weights = create_cpg_structure(self.num_osc)
        self.cpg = CPG(weights=self.weights, dt=control_timestep, solver=RK4Solver())
        self.cpg_state = None

    def _initialize_gait(self, cpg: CPG, env: environment.Environment) -> Any:
        """Sets up the modulated rowing gait state."""
        self.cpg_state = cpg.reset(rng=jax.random.PRNGKey(0))

        # Set base frequency
        cpg_state = cpg_state.replace(omegas=jnp.pi / 2 * jnp.ones_like(cpg_state.omegas))  # type: ignore

        # Define rowing roles
        leading_idx = 0
        left_rowers = [(leading_idx - 1) % NUM_ARMS, (leading_idx - 2) % NUM_ARMS]
        right_rowers = [(leading_idx + 1) % NUM_ARMS, (leading_idx + 2) % NUM_ARMS]

        return modulate_rowing_gait(
            cpg_state=cpg_state,
            leading_arms=[leading_idx],
            left_rowers=left_rowers,
            right_rowers=right_rowers,
            left_second=[left_rowers[1]],
            right_second=[right_rowers[1]],
            max_joint_limit=env.env.action_space.high[0] * 0.5,  # type: ignore
        )

    def __call__(self, env: environment.Environment, actions: jnp.ndarray) -> BaseEnvState:
        next_cpg_state = self.cpg.step(self.cpg_state)

        cpg_outputs_per_arm = next_cpg_state.outputs.reshape((NUM_ARMS, 2))
        actions = jnp.repeat(cpg_outputs_per_arm, NUM_SEGMENTS, axis=0).ravel()

        # TODO: jit compile step
        next_env_state = env.env.step(env.state, actions)

        self.cpg_state = next_cpg_state

        return next_env_state
