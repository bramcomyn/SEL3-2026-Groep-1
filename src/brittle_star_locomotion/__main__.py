import argparse

from brittle_star_locomotion.config.configuration import Configuration
from brittle_star_locomotion.logger.logger import Logger

def main():
    """Main entry point for the brittle star locomotion project."""
    arguments = _parse_arguments()

    configuration = Configuration(arguments.config).configuration
    logger = Logger(name=__name__, verbose=arguments.verbose)

    logger.info(f"Starting brittle star locomotion project with configuration: {configuration}")
    logger.info(f"Running in {arguments.mode} mode.")

    mode_dictionary[arguments.mode]()

    logger.info("Finished brittle star locomotion project.")

def train():
    """Train the brittle star locomotion model."""
    logger = Logger()
    logger.debug("Starting training process...")

    logger.debug("Training process completed.")

def evaluate():
    """Evaluate the brittle star locomotion model."""
    logger = Logger()
    logger.debug("Starting evaluation process...")

    logger.debug("Evaluation process completed.")

def _parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Brittle Star Locomotion")

    parser.add_argument("-c", "--config",  type=str, default="configs/base_config.yaml",         help="path to the configuration file")
    parser.add_argument("-v", "--verbose", action="store_true",                                  help="enable verbose logging")
    parser.add_argument("-m", "--mode",    type=str, choices=mode_dictionary.keys(), default="train", help="mode to run the project in (training or evaluation)")

    return parser.parse_args()

# mapping of mode strings to functions
mode_dictionary = {
    "train": train,
    "eval": evaluate,
}

if __name__ == "__main__":
    main()
