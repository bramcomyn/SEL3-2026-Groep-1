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
    parser.add_argument("-d", "--debug", dest="loglevel", action="store_const", const=logging.DEBUG)
    parser.add_argument("-f", "--config-file", dest="config_file", type=str, default="configs/base_config.yaml")
    parser.add_argument("-t", "--train", dest="train", action="store_true", default=False)
    parser.add_argument("-v", "--verbose", dest="loglevel", action="store_const", const=logging.INFO, default=logging.INFO)
    parser.add_argument("--output", type=str, default="out/brittle_star_sim.mp4")
    return parser.parse_args()


def main():
    args = get_args()

    logging.basicConfig(level=args.loglevel, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)

    logger = logging.getLogger(__name__)
    config = load_config(args.config_file)
    if not args.train: # if not training, we don't need 1000s of environments
        logging.info("Falling back to single environment.")
        config.rl.amount_environments = 1

    obs_to_use = config.rl.observations_to_use
    env = Environment(config, observations=obs_to_use)

    n_agents = config.env.num_arms
    if args.train:
        # Initialize IQL Trainer
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.rl.gradient_clip),
            optax.adam(config.rl.learning_rate)
        )

        trainer = IndependentQLearning(optimizer=optimizer, n_agents=n_agents, env=env)

        # Training Phase
        logger.info("Starting Training...")

        trainer.train()
        trainer.save(args.checkpoint)
        networks = trainer.value_networks

        logger.info("Training complete.")
    else:
        networks = [
            load_checkpoint(
                lambda: QNetwork(
                    len(obs_to_use),
                    config.rl.action_space_dim,
                    nnx.Rngs(i + config.env.num_arms),
                    config.rl.hidden_layer_size, 
                    config.rl.amount_of_hidden_layers
                ),
                f"{args.checkpoint}_{i}"
            ) for i in range(config.env.num_arms)
        ]

    # Visualization / Evaluation Phase
    if config.evaluation.render and not args.train:
        logger.info("Running evaluation for visualization...")
        env.reset()
        eval_trajectory = []

        num_eval_cycles = 20
        for _ in tqdm(range(num_eval_cycles), desc="Evaluation cycles"):
            # get_observations() returns (num_envs, num_arms, obs_dim)
            observations = env.get_observations()

            # shape: (num_envs, num_arms)
            all_env_actions = []
            for e in range(config.rl.amount_environments):
                q_values = jnp.stack(
                    [networks[agent](observations[e, agent]) for agent in range(n_agents)], 
                    axis=0
                )

                all_env_actions.append(jnp.argmax(q_values, axis=1))
            
            actions = jnp.stack(all_env_actions, axis=0)

            # Use the standard step() method
            *_, trajectory = env.step(actions)
            eval_trajectory.append(trajectory)

            if jnp.any(env.env_state.terminated):
                logger.info("Environment terminated during evaluation. Resetting environment.")
                env.reset()

        # turning the (20, 1, 50) into (20, 50)
        combined_trajectory = jax.tree_util.tree_map(
            lambda *xs: jnp.concatenate(xs, axis=1),
            *eval_trajectory
        )

        output_path = config.evaluation.output_video_path
        env.render_video(combined_trajectory, output_path=f"{output_path}-{time.strftime("%Y%m%d-%H%M%S")}.mp4")
        logger.info(f"Video saved to {output_path}")


if __name__ == "__main__":
    main()
