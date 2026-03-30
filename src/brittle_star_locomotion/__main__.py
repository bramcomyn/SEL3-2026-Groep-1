import argparse
import logging
import time

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from tqdm import tqdm

from brittle_star_locomotion.config.config_loader import load_config
from brittle_star_locomotion.environment import Environment
from brittle_star_locomotion.neural.checkpoint import load_checkpoint
from brittle_star_locomotion.neural.qnetwork import QNetwork
from brittle_star_locomotion.optimization.independentqlearning import IndependentQLearning


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Brittle Star Locomotion Simulator")
    parser.add_argument("-c", "--checkpoint", dest="checkpoint", default="checkpoint")
    parser.add_argument("-t", "--train", dest="train", action="store_true")
    parser.add_argument("-v", "--verbose", dest="loglevel", action="store_const", const=logging.INFO, default=logging.INFO)
    parser.add_argument("-d", "--debug", dest="loglevel", action="store_const", const=logging.DEBUG)
    parser.add_argument("--output", type=str, default="out/brittle_star_sim.mp4")
    return parser.parse_args()


def main():
    args = get_args()

    logging.basicConfig(level=args.loglevel, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)

    logger = logging.getLogger(__name__)
    config = load_config("configs/base_config.yaml")

    # Initialize Environment
    obs_to_use = config.rl.observations_to_use
    env = Environment(observations=obs_to_use)

    n_agents = config.env.num_arms
    if args.train:
        # Initialize IQL Trainer
        learning_rate = config.rl.learning_rate
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.rl.gradient_clip),
            optax.adam(learning_rate)
        )

        trainer = IndependentQLearning(optimizer=optimizer, n_agents=n_agents, env=env)

        # Training Phase
        logger.info("Starting Training...")

        trainer.train()
        trainer.save(args.checkpoint)
        networks = trainer.value_networks

        logger.info("Training complete.")
    else:
        networks = [load_checkpoint(
            lambda: QNetwork(
                len(obs_to_use),
                config.rl.action_space_dim,
                nnx.Rngs(i + config.env.num_arms),
                config.rl.hidden_layer_size, 
                config.rl.amount_of_hidden_layers
            ),
            f"{args.checkpoint}_{i}"
        ) for i in range(config.env.num_arms)]

    # Visualization / Evaluation Phase
    if config.evaluation.render:
        logger.info("Running evaluation for visualization...")
        env.reset()
        eval_trajectory = []
        
        # Run for a fixed number of cycles to generate a video
        num_eval_cycles = 20
        for _ in tqdm(range(num_eval_cycles), desc="Evaluation cycles"):
            observations = env.get_observations()

            # Get greedy actions from the trained network
            # (n_agents, action_probs) -> (n_agents,)
            q_values = jnp.stack(
                [networks[agent](observations[agent]) for agent in range(n_agents)], 
                axis=0
            )

            actions = jnp.argmax(q_values, axis=1)

            # run_iteration returns the trajectory of MJX states
            trajectory = env.run_iteration(actions)
            eval_trajectory.append(trajectory)

            if env.env_state.terminated:
                logger.info("Environment terminated during evaluation. Resetting environment.")
                env.reset()

        # 6. Combine and Render
        combined_trajectory = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *eval_trajectory)

        output_path = config.evaluation.output_video_path
        env.render_video(combined_trajectory, output_path=f"{output_path}-{time.strftime("%Y%m%d-%H%M%S")}.mp4")
        logger.info(f"Video saved to {output_path}")


if __name__ == "__main__":
    main()
