import argparse
import os
import time

import jax
import jax.numpy as jnp
from flax import nnx

from brittle_star_locomotion.config.configuration import Configuration
from brittle_star_locomotion.environment.environment import Environment
from brittle_star_locomotion.environment.render import EnvironmentRenderer
from brittle_star_locomotion.logger.logger import Logger
from brittle_star_locomotion.neural.qnetwork import QNetwork
from brittle_star_locomotion.optimizer.iql_optimizer import IQLOptimizer

def main():
    """Main entry point for the brittle star locomotion project."""
    arguments = _parse_arguments()

    _ = Configuration(arguments.config).configuration
    logger = Logger(name=__name__, verbose=arguments.verbose)

    logger.info(f"Starting brittle star locomotion project.")
    logger.info(f"Running in {arguments.mode} mode.")

    mode_dictionary[arguments.mode](arguments)

    logger.info("Finished brittle star locomotion project.")

def train(arguments: argparse.Namespace):
    """Train the brittle star locomotion model."""
    logger = Logger()
    started_at = time.perf_counter()
    logger.info("Starting training process...")

    environment = Environment()
    optimizer = IQLOptimizer(environment)
    
    logger.info(f"Training with {environment.number_of_environments} parallel environments")
    optimizer.optimize()
    
    checkpoint_base = _normalize_checkpoint_base_name(arguments.checkpoint, Configuration().configuration.checkpoint_directory)
    optimizer.save_model(checkpoint_prefix=checkpoint_base)
    
    elapsed = time.perf_counter() - started_at
    logger.info(f"Training completed in {elapsed:.1f}s")
    logger.debug("Training process finished.")

def evaluate(arguments: argparse.Namespace):
    """Evaluate the brittle star locomotion model."""
    logger = Logger()
    logger.debug("Starting evaluation process...")
    started_at = time.perf_counter()

    config = Configuration().configuration

    environment = Environment()
    checkpoint_base = _normalize_checkpoint_base_name(arguments.checkpoint, config.checkpoint_directory)
    q_networks = _load_qnetworks_for_evaluation(
        environment=environment,
        checkpoint_base=checkpoint_base,
        logger=logger,
    )

    render_trajectory, actions_trajectory = _collect_trajectory(
        environment,
        q_networks,
        num_modulation_steps=config.gait.fixed_number_of_evaluation_modulation_steps,
        num_substeps=config.gait.fixed_number_of_evaluation_substeps_per_modulation,
        log_every=config.environment.render_every_x_frames,
        logger=logger,
    )

    renderer = EnvironmentRenderer(environment)
    renderer.render_video(render_trajectory, output_path=arguments.output_video)
    logger.info(f"Saved evaluation video to: {arguments.output_video}")

    _save_action_trajectory(arguments.output_trajectory, actions_trajectory)
    logger.info("Saving action trajectory")

    elapsed = time.perf_counter() - started_at
    logger.info(f"Evaluation completed in {elapsed:.1f}s")
    logger.debug("Evaluation process completed.")

def _normalize_checkpoint_base_name(checkpoint_argument: str, checkpoint_directory: str) -> str:
    """Convert checkpoint CLI input to a basename used by QNetwork.load_checkpoint."""
    normalized = checkpoint_argument.replace("\\", "/")
    checkpoint_directory = checkpoint_directory.rstrip("/")

    if os.path.isabs(normalized):
        return os.path.basename(normalized)

    if normalized.startswith(f"{checkpoint_directory}/"):
        normalized = normalized[len(checkpoint_directory) + 1 :]

    normalized = normalized.split("/")[-1]

    return normalized

def _resolve_agent_checkpoint_name(checkpoint_base: str, agent_id: int, checkpoint_directory: str) -> str:
    """Resolve the checkpoint filename for a specific agent."""
    checkpoint_name = f"{checkpoint_base}_{agent_id}"
    checkpoint_path = os.path.join(checkpoint_directory, checkpoint_name)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint for agent {agent_id} was not found at {checkpoint_path}. "
            f"Expected naming pattern: <checkpoint_prefix>_<agent_id>."
        )

    return checkpoint_name

def _load_qnetworks_for_evaluation(
    environment: Environment,
    checkpoint_base: str,
    logger: Logger,
) -> list[QNetwork]:
    """Load one Q-network checkpoint per agent for greedy policy evaluation."""
    config = Configuration().configuration
    observation_size = environment.get_observation_size()
    checkpoint_directory = config.checkpoint_directory
    q_networks = []

    for agent_id in range(config.environment.number_of_arms):
        checkpoint_name = _resolve_agent_checkpoint_name(
            checkpoint_base=checkpoint_base,
            agent_id=agent_id,
            checkpoint_directory=checkpoint_directory,
        )

        logger.info(f"Loading checkpoint for agent {agent_id}: {checkpoint_name}")

        q_network = QNetwork.load_checkpoint(
            model_callback=lambda agent_id=agent_id: QNetwork(
                input_size=observation_size,
                output_size=5,
                rngs=nnx.Rngs(agent_id),
                hidden_size=config.rl.hidden_size,
                amount_of_hidden_layers=config.rl.amount_of_hidden_layers,
            ),
            name=checkpoint_name,
        )

        q_networks.append(q_network)

    return q_networks

def _select_policy_actions(
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
    environment: Environment,
    q_networks: list[QNetwork],
    num_modulation_steps: int,
    num_substeps: int,
    log_every: int,
    logger: Logger,
):
    """Run a single greedy-policy episode and stack the renderable environment trajectory."""
    env_state, cpg_state = environment.reset()
    logger.info(
        f"Starting policy rollout with {num_modulation_steps} modulation steps and {num_substeps} substeps per step"
    )

    done_environments = jnp.zeros((environment.number_of_environments,), dtype=bool)
    trajectory_env_states_list = []
    trajectory_actions_list = []

    for step_index in range(num_modulation_steps):
        if bool(jnp.all(done_environments)):
            break

        if step_index % max(1, log_every) == 0 or step_index == num_modulation_steps - 1:
            logger.info(f"Evaluation progress: modulation step {step_index + 1}/{num_modulation_steps}")

        observations = environment.get_observations()
        actions = _select_policy_actions(observations=observations, q_networks=q_networks)
        actions = jnp.where(done_environments[:, None], 0, actions) # (n_environments, n_agents)

        env_state, cpg_state, _, terminated, truncated, trajectory = environment.step(
            env_state,
            cpg_state,
            actions,
            num_substeps,
        )

        terminated = jnp.asarray(terminated).reshape(-1)
        truncated = jnp.asarray(truncated).reshape(-1)
        done_environments = done_environments | terminated | truncated

        environment.env_state = env_state
        environment.cpg_state = cpg_state

        substep_env_states = trajectory[0]

        trajectory_env_states_list.append(substep_env_states)
        trajectory_actions_list.append(actions)

    logger.info(
        f"Policy rollout finished after {len(trajectory_env_states_list)} modulation steps"
    )

    if not (trajectory_env_states_list and trajectory_actions_list):
        raise ValueError("No trajectory data collected during evaluation")

    trajectory_env_states = jax.tree_util.tree_map(
        lambda *xs: jnp.concatenate(xs, axis=0),
        *trajectory_env_states_list,
    )
    trajectory_env_states = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), trajectory_env_states)
    trajectory_actions = jnp.concatenate(trajectory_actions_list, axis=0)

    return trajectory_env_states, trajectory_actions

def _save_action_trajectory(output_filename: str, action_trajectory: jnp.ndarray) -> None:
    if len(action_trajectory.shape) == 3:
        n_steps, n_environments, n_agents = action_trajectory.shape
    else:
        n_steps, n_agents = action_trajectory.shape
        n_environments = 1
        action_trajectory = action_trajectory[:, None, :]

    with open(f'out/{output_filename}', 'w') as output:
        output.write(f'environment_id,step_id,agent_id,action\n')

        for environment_id in range(n_environments):
            for step_id in range(n_steps):
                for agent_id in range(n_agents):
                    output.write(f'{environment_id},{step_id},{agent_id},{action_trajectory[step_id, environment_id, agent_id]}\n')

def _parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Brittle Star Locomotion")

    parser.add_argument("-c", "--config",          type=str, default="configs/base_config.yaml",               help="path to the configuration file")
    parser.add_argument("-v", "--verbose",         action="store_true",                                        help="enable verbose logging")
    parser.add_argument("-m", "--mode",            type=str, choices=mode_dictionary.keys(), default="train",  help="mode to run the project in (training or evaluation)")
    parser.add_argument("-p", "--checkpoint",      type=str, default="checkpoints/test_checkpoint",            help="path to the model checkpoint for evaluation (prefix for the checkpoint files)")
    parser.add_argument("--output-video",          type=str, default="out/eval.mp4",                           help="path to save evaluation video")
    parser.add_argument("--output-trajectory",     type=str, default="out/eval.csv",                           help="path to save action trajectory csv")

    return parser.parse_args()

# mapping of mode strings to functions
mode_dictionary = {
    "train": train,
    "eval": evaluate,
}

if __name__ == "__main__":
    main()
