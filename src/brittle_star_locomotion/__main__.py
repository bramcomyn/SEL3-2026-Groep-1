import argparse
import logging
import sys

from brittle_star_locomotion.core import run_experiment, visualize_agent


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Brittle Star Locomotion Simulator")

    parser.add_argument("-v", "--verbose", help="Increase output verbosity (INFO)", action="store_const", const=logging.INFO, default=logging.WARNING)

    parser.add_argument("-d", "--debug", help="Show debug/JIT trace logs", action="store_const", const=logging.DEBUG)

    parser.add_argument("--time", type=float, default=20.0, help="Simulation time in seconds")
    parser.add_argument("--render", action="store_true", help="Render video output")

    return parser.parse_args()


def get_logger(args: argparse.Namespace) -> logging.Logger:
    log_level = args.debug or args.verbose

    logging.basicConfig(level=log_level, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    return logging.getLogger("brittle_star")


def main():
    args = get_args()
    logger = get_logger(args)

    logger.info(f"Starting simulation")

    # run_experiment(simulation_time=args.time)
    visualize_agent("test_checkpoint", args.time)


if __name__ == "__main__":
    main()
