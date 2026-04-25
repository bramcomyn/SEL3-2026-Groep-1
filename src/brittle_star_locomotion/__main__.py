import argparse

from brittle_star_locomotion.config.configuration import Configuration
from brittle_star_locomotion.logger.logger import Logger
from brittle_star_locomotion.evaluation import Evaluator
from brittle_star_locomotion.training import Trainer

def main():
    """Main entry point for the brittle star locomotion project."""
    arguments = _parse_arguments()

    _ = Configuration(arguments.config).configuration
    logger = Logger(name=__name__, verbose=arguments.verbose)

    logger.info("Starting brittle star locomotion project.")
    logger.info(f"Running in {arguments.mode} mode.")

    mode_dictionary[arguments.mode](arguments)

    logger.info("Finished brittle star locomotion project.")

def _parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Brittle Star Locomotion")

    parser.add_argument("-c", "--config",          type=str, default="configs/base_config.yaml",               help="path to the configuration file")
    parser.add_argument("-v", "--verbose",         action="store_true",                                        help="enable verbose logging")
    parser.add_argument("-m", "--mode",            type=str, choices=mode_dictionary.keys(), default="train",  help="mode to run the project in (training or evaluation)")
    parser.add_argument("-p", "--checkpoint",      type=str, default="checkpoints/test_checkpoint",            help="path to the model checkpoint for evaluation (prefix for the checkpoint files)")
    parser.add_argument("--output-video",          type=str, default="out/eval.mp4",                           help="path to save evaluation video")
    parser.add_argument("--render",                action="store_true",                                        help="render the evaluation trajectory")
    parser.add_argument("--output-actions-trajectory",       type=str, default="out/eval_actions.csv",         help="path to save action trajectory csv")
    parser.add_argument("--output-positions-trajectory",     type=str, default="out/eval_positions.csv",       help="path to save positions trajectory csv")

    return parser.parse_args()

# mapping of mode strings to functions
mode_dictionary = {
    "train": Trainer().train,
    "eval": Evaluator().evaluate,
}

if __name__ == "__main__":
    main()
