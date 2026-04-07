import argparse
import time

import jax
import jax.numpy as jnp

from brittle_star_locomotion.config.configuration import Configuration
from brittle_star_locomotion.environment.environment import Environment
from brittle_star_locomotion.environment.render import EnvironmentRenderer
from brittle_star_locomotion.logger.logger import Logger

def main():
    """Main entry point for the brittle star locomotion project."""
    arguments = _parse_arguments()

    configuration = Configuration(arguments.config).configuration
    logger = Logger(name=__name__, verbose=arguments.verbose)

    logger.info(f"Starting brittle star locomotion project.")
    logger.info(f"Running in {arguments.mode} mode.")

    mode_dictionary[arguments.mode](arguments)

    logger.info("Finished brittle star locomotion project.")

def train(_arguments: argparse.Namespace):
    """Train the brittle star locomotion model."""
    logger = Logger()
    logger.debug("Starting training process...")

    # TODO: Implement the training logic here

    logger.debug("Training process completed.")

def evaluate(arguments: argparse.Namespace):
    """Evaluate the brittle star locomotion model."""
    logger = Logger()
    logger.debug("Starting evaluation process...")
    started_at = time.perf_counter()

    config = Configuration().configuration
    config.rl.number_of_environments = 1
    config.env.render_every = arguments.eval_render_every

    fixed_action = _parse_fixed_action(arguments.eval_fixed_action)
    if fixed_action.shape[0] != config.env.num_arms:
        raise ValueError(
            f"Expected {config.env.num_arms} action entries, got {fixed_action.shape[0]}"
        )

    environment = Environment()
    render_trajectory = _collect_render_trajectory(
        environment,
        fixed_action,
        num_modulation_steps=arguments.eval_modulation_steps,
        num_substeps=arguments.eval_substeps,
        log_every=arguments.eval_log_every,
        logger=logger,
    )

    renderer = EnvironmentRenderer(environment)
    renderer.render_video(render_trajectory, output_path=arguments.output_video)

    elapsed = time.perf_counter() - started_at
    logger.info(f"Saved evaluation video to: {arguments.output_video}")
    logger.info(f"Evaluation completed in {elapsed:.1f}s")

    logger.debug("Evaluation process completed.")


def _parse_fixed_action(value: str) -> jnp.ndarray:
    """Parse a comma-separated list of arm roles into an integer JAX array."""
    entries = [item.strip() for item in value.split(",") if item.strip()]
    if not entries:
        raise ValueError("--eval-fixed-action must contain at least one role value")
    return jnp.array([int(x) for x in entries], dtype=jnp.int32)


def _collect_render_trajectory(
    environment: Environment,
    fixed_action: jnp.ndarray,
    num_modulation_steps: int,
    num_substeps: int,
    log_every: int,
    logger: Logger,
):
    """Run repeated fixed-action rollouts and stack the renderable environment trajectory."""
    env_state, cpg_state = environment.reset()
    logger.info(
        f"Starting fixed-action rollout with {num_modulation_steps} modulation steps and {num_substeps} substeps per step"
    )

    trajectory_env_states_list = []
    for step_index in range(num_modulation_steps):
        if step_index % max(1, log_every) == 0 or step_index == num_modulation_steps - 1:
            logger.info(f"Evaluation progress: modulation step {step_index + 1}/{num_modulation_steps}")

        env_state, cpg_state, _, _, _, trajectory = environment.step(
            env_state,
            cpg_state,
            fixed_action,
            num_substeps,
        )

        substep_env_states = trajectory[0]

        trajectory_env_states_list.append(substep_env_states)

    if not trajectory_env_states_list:
        raise ValueError("No trajectory data collected during evaluation")

    trajectory_env_states = jax.tree_util.tree_map(
        lambda *xs: jnp.concatenate(xs, axis=0),
        *trajectory_env_states_list,
    )

    return jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), trajectory_env_states)

def _parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Brittle Star Locomotion")

    parser.add_argument("-c", "--config",          type=str, default="configs/base_config.yaml",               help="path to the configuration file")
    parser.add_argument("-v", "--verbose",         action="store_true",                                        help="enable verbose logging")
    parser.add_argument("-m", "--mode",            type=str, choices=mode_dictionary.keys(), default="train",  help="mode to run the project in (training or evaluation)")
    parser.add_argument("-p", "--checkpoint",      type=str, default="checkpoints/test_checkpoint",            help="path to the model checkpoint for evaluation (prefix for the checkpoint files)")
    parser.add_argument("--output-video",          type=str, default="out/eval.mp4",                           help="path to save evaluation video")
    parser.add_argument("--eval-modulation-steps", type=int, default=20,                                       help="number of modulation steps to run during evaluation")
    parser.add_argument("--eval-substeps",         type=int, default=50,                                       help="number of CPG/physics substeps per modulation step")
    parser.add_argument("--eval-render-every",     type=int, default=1,                                        help="frame stride used by the evaluation renderer")
    parser.add_argument("--eval-log-every",        type=int, default=1,                                        help="log progress every N modulation steps")
    parser.add_argument(
        "--eval-fixed-action",
        type=str,
        default="0,4,3,1,2",
        help="comma-separated arm role assignment used each modulation step",
    )

    return parser.parse_args()

# mapping of mode strings to functions
mode_dictionary = {
    "train": train,
    "eval": evaluate,
}

if __name__ == "__main__":
    main()
