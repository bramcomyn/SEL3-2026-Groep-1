import argparse
import jax
import logging
import optax
import jax.numpy as jnp

from brittle_star_locomotion.environment import Environment, NUM_ARMS
from brittle_star_locomotion.optimization.independentqlearning import IndependentQLearning


NUM_MODULATIONS = 10


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Brittle Star Locomotion Simulator")
    parser.add_argument("-v", "--verbose", dest="loglevel", action="store_const", const=logging.INFO, default=logging.INFO)
    parser.add_argument("-d", "--debug", dest="loglevel", action="store_const", const=logging.DEBUG)
    parser.add_argument("--output", type=str, default="out/brittle_star_sim.mp4")
    return parser.parse_args()


def main():
    args = get_args()

    logging.basicConfig(level=args.loglevel, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)

    logger = logging.getLogger(__name__)

    # 2. Initialize Environment
    # Note: Using subset of observations to keep state space manageable
    obs_to_use = ["angle_to_target"]
    env = Environment(observations=obs_to_use)

    # 3. Initialize IQL Trainer
    n_agents = NUM_ARMS
    learning_rate = 0.0001
    optimizer = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(learning_rate))

    trainer = IndependentQLearning(optimizer=optimizer, n_agents=n_agents, env=env, replay_buffer_size=1000)

    # 4. Training Phase
    logger.info("Starting Training...")

    trainer.train(n_episodes=50, epsilon=1.0, epsilon_decay=0.95, epsilon_min=0.01, batch_size=32, discount=0.99)

    logger.info("Training complete.")

    # 5. Visualization / Evaluation Phase
    # We run one final episode with epsilon=0 (greedy) to see the result
    logger.info("Running evaluation for visualization...")
    env.reset()
    eval_trajectory = []
    
    # Run for a fixed number of cycles to generate a video
    num_eval_cycles = 20
    for i in range(num_eval_cycles):
        observations = env.get_observations()

        # Get greedy actions from the trained network
        # (n_agents, action_probs) -> (n_agents,)
        q_values = trainer.value_network(observations)
        actions = jnp.argmax(q_values, axis=1)

        # run_iteration returns the trajectory of MJX states
        trajectory = env.run_iteration(actions)
        eval_trajectory.append(trajectory)

        if i % 5 == 0:
            logger.info(f"Eval Cycle {i}/{num_eval_cycles}")

    # 6. Combine and Render
    combined_trajectory = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *eval_trajectory)

    output_path = "out/trained_brittle_star.mp4"
    env.render_video(combined_trajectory, output_path=output_path)
    logger.info(f"Video saved to {output_path}")


if __name__ == "__main__":
    main()
