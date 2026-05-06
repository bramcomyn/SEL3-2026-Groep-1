import argparse
import time

import jax
import jax.numpy as jnp
from flax import nnx

from brittle_star_locomotion.config.configuration import Configuration
from brittle_star_locomotion.environment.fixedtargetenvironment import FixedTargetEnvironment
from brittle_star_locomotion.environment.randomtargetenvironment import RandomTargetEnvironment
from brittle_star_locomotion.environment.render import EnvironmentRenderer
from brittle_star_locomotion.logger.logger import Logger
from brittle_star_locomotion.neural.qnetwork import QNetwork
from brittle_star_locomotion.util.checkpoint_naming import normalize_checkpoint_base_name, resolve_agent_checkpoint_name
from brittle_star_locomotion.damage.arm_damage import ArmDamage

class Evaluator:
    def __init__(self):
        self.logger = Logger()
        self.config = Configuration().configuration
        self.environment = RandomTargetEnvironment() if self.config.environment.random_target else FixedTargetEnvironment()

    def evaluate(self, arguments: argparse.Namespace):
        """Evaluate the brittle star locomotion model.
        This method loads the trained Q-networks for each agent, runs a greedy policy rollout in the environment, and collects the trajectory data.
        The collected trajectory can be rendered to a video and saved to disk, along with the action and position trajectories in CSV format.
        
        :param arguments: Command-line arguments containing configuration and checkpoint information.
        """

        self.logger.debug("Starting evaluation process...")
        started_at = time.perf_counter()

        checkpoint_base = normalize_checkpoint_base_name(arguments.checkpoint, self.config.checkpoint_directory)
        q_networks = self._load_qnetworks_for_evaluation(
            checkpoint_base=checkpoint_base,
        )

        render_trajectory, actions_trajectory, positions_trajectory, breakpoints_trajectory, broken_arms_trajectory = self._collect_trajectory(
            q_networks,
            num_modulation_steps=self.config.gait.fixed_number_of_evaluation_modulation_steps,
            num_substeps=self.config.gait.fixed_number_of_evaluation_substeps_per_modulation,
            log_every=self.config.environment.render_every_x_frames,
        )

        if arguments.render:
            renderer = EnvironmentRenderer(self.environment)
            renderer.render_video(render_trajectory, output_path=arguments.output_video)
            self.logger.info(f"Saved evaluation video to: {arguments.output_video}")

        self._save_action_trajectory(arguments.output_actions_trajectory, actions_trajectory)
        self.logger.info("Saving action trajectory")

        self._save_position_trajectory(arguments.output_positions_trajectory, positions_trajectory, self.environment.target_position)
        self.logger.info("Saving position trajectory")

        self._save_breakpoint_trajectory(arguments.output_breakpoints_trajectory, breakpoints_trajectory, broken_arms_trajectory)
        self.logger.info("Saving breakpoint trajectory")

        elapsed = time.perf_counter() - started_at
        self.logger.info(f"Evaluation completed in {elapsed:.1f}s")
        self.logger.debug("Evaluation process completed.")

    def _load_qnetworks_for_evaluation(
        self,
        checkpoint_base: str,
    ) -> list[QNetwork]:
        """Load one Q-network checkpoint per agent for greedy policy evaluation."""
        observation_size = self.environment.get_observation_size()
        checkpoint_directory = self.config.checkpoint_directory
        q_networks = []

        for agent_id in range(self.config.environment.number_of_arms):
            checkpoint_name = resolve_agent_checkpoint_name(
                checkpoint_base=checkpoint_base,
                agent_id=agent_id,
                checkpoint_directory=checkpoint_directory,
            )

            self.logger.info(f"Loading checkpoint for agent {agent_id}: {checkpoint_name}")

            q_network = QNetwork.load_checkpoint(
                model_callback=lambda agent_id=agent_id: QNetwork(
                    input_size=observation_size,
                    output_size=5,
                    rngs=nnx.Rngs(agent_id),
                    hidden_size=self.config.rl.hidden_size,
                    amount_of_hidden_layers=self.config.rl.amount_of_hidden_layers,
                ),
                name=checkpoint_name,
            )

            q_networks.append(q_network)

        return q_networks

    def _select_policy_actions(
        self,
        observations: jnp.ndarray,
        q_networks: list[QNetwork],
    ) -> jnp.ndarray:
        """Select greedy actions from the loaded per-agent Q-networks."""
        q_values_by_agent = []
        for agent_id, q_network in enumerate(q_networks):
            q_values_by_agent.append(q_network(observations[:, agent_id, :]))

        q_values = jnp.stack(q_values_by_agent, axis=1)
        return jnp.argmax(q_values, axis=-1).astype(jnp.int32)

    def _collect_trajectory(
        self,
        q_networks: list[QNetwork],
        num_modulation_steps: int,
        num_substeps: int,
        log_every: int,
    ):
        """Run a single greedy-policy episode and stack the renderable environment trajectory."""
        env_state, cpg_state = self.environment.reset()
        self.logger.info(
            f"Starting policy rollout with {num_modulation_steps} modulation steps and {num_substeps} substeps per step"
        )

        done_environments = jnp.zeros((self.environment.number_of_environments,), dtype=bool)
        trajectory_env_states_list = []
        trajectory_actions_list = []
        trajectory_positions_list = []

        arm_damage = ArmDamage()

        for step_idx in range(num_modulation_steps):
            if bool(jnp.all(done_environments)):
                break

            if step_idx % max(1, log_every) == 0 or step_idx == num_modulation_steps - 1:
                self.logger.info(f"Evaluation progress: modulation step {step_idx + 1}/{num_modulation_steps}")

            observations = self.environment.get_observations()
            actions = self._select_policy_actions(observations=observations, q_networks=q_networks)
            actions = jnp.where(done_environments[:, None], 0, actions) # (n_environments, n_agents)
    
            arm_damage.break_arms(step_idx)

            env_state, cpg_state, _, terminated, truncated, trajectory = self.environment.step(
                env_state,
                cpg_state,
                actions,
                arm_damage.get_active_arms(),
                num_substeps,
            )

            terminated = jnp.asarray(terminated).reshape(-1)
            truncated = jnp.asarray(truncated).reshape(-1)
            done_environments = done_environments | terminated | truncated

            self.environment.env_state = env_state
            self.environment.cpg_state = cpg_state

            substep_env_states = trajectory[0]

            trajectory_env_states_list.append(substep_env_states)
            trajectory_actions_list.append(actions)
            trajectory_positions_list.append(self.environment.env_state.observations["disk_position"])

        self.logger.info(
            f"Policy rollout finished after {len(trajectory_env_states_list)} modulation steps"
        )

        if not (trajectory_env_states_list and trajectory_actions_list):
            raise ValueError("No trajectory data collected during evaluation")

        trajectory_env_states = jax.tree_util.tree_map(
            lambda *xs: jnp.concatenate(xs, axis=0),
            *trajectory_env_states_list,
        )
        trajectory_env_states = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), trajectory_env_states)
        trajectory_actions = jnp.stack(trajectory_actions_list, axis=1)
        trajectory_positions = jnp.stack(trajectory_positions_list, axis=1)
        trajectory_breakpoints = arm_damage._break_points

        # https://chatgpt.com/share/69fb4a9f-8210-8326-acf5-90d90975f1e7
        _, trajectory_broken_arms = jnp.where(
            arm_damage.get_active_arms() == 0 # shape (n_envs, n_agents)
        ) # shape (n_envs,)

        return (
            trajectory_env_states, 
            trajectory_actions, 
            trajectory_positions, 
            trajectory_breakpoints,
            trajectory_broken_arms
        )

    def _save_action_trajectory(self, output_filename: str, action_trajectory: jnp.ndarray) -> None:
        if len(action_trajectory.shape) == 3:
            n_environments, n_steps, n_agents = action_trajectory.shape
        else:
            n_steps, n_agents = action_trajectory.shape
            n_environments = 1
            action_trajectory = action_trajectory[:, None, :]

        with open(f'{output_filename}', 'w') as output:
            output.write('environment_id,step_id,agent_id,action\n')

            for environment_id in range(n_environments):
                for step_id in range(n_steps):
                    for agent_id in range(n_agents):
                        output.write(f'{environment_id},{step_id},{agent_id},{action_trajectory[step_id, environment_id, agent_id]}\n')

    def _save_position_trajectory(self, output_filename: str, positions_trajectory: jnp.ndarray, target_positions: jnp.ndarray) -> None:
        """ positions_trajectory - (n_environments, n_steps, 3) or (n_steps, 3)
        """
        if len(positions_trajectory.shape) == 3:
            n_environments, n_steps, _ = positions_trajectory.shape
        else:
            n_steps, _ = positions_trajectory.shape
            n_environments = 1
            positions_trajectory = positions_trajectory[None, :, :]

        with open(f'{output_filename}', 'w') as output:
            output.write(f'environment_id,step_id,x,y,in_trajectory\n')

            for environment_id in range(n_environments): 
                output.write(f'{environment_id},0,0,0,true\n') # Start position

                for step_id in range(n_steps):
                    x = positions_trajectory[environment_id, step_id, 0]
                    y = positions_trajectory[environment_id, step_id, 1]
                    output.write(f'{environment_id},{step_id},{x},{y},true\n')

                end_x, end_y, _ = tuple(target_positions[environment_id])
                output.write(f'{environment_id},{n_steps},{end_x},{end_y},false\n') # End position

    def _save_breakpoint_trajectory(self, output_filename: str, breakpoint_trajectory: jnp.ndarray, broken_arms_trajectory: jnp.ndarray) -> None:
        """ breakpoint_trajectory - (n_environments,)
        """
        n_environments = breakpoint_trajectory.shape[0]

        print(broken_arms_trajectory)

        with open(f'{output_filename}', 'w') as output:
            output.write(f'environment_id,breakpoint,agent_id\n')

            for environment_id in range(n_environments):
                output.write(f'{environment_id},{breakpoint_trajectory[environment_id]},{broken_arms_trajectory[environment_id]}\n')
