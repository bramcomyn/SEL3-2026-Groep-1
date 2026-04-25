import argparse
import time

from brittle_star_locomotion.config.configuration import Configuration
from brittle_star_locomotion.environment.fixedtargetenvironment import FixedTargetEnvironment
from brittle_star_locomotion.environment.randomtargetenvironment import RandomTargetEnvironment
from brittle_star_locomotion.logger.logger import Logger
from brittle_star_locomotion.optimizer.iql_optimizer import IQLOptimizer
from brittle_star_locomotion.util.checkpoint_naming import normalize_checkpoint_base_name

class Trainer:
    def __init__(self):
        self.logger = Logger()
        self.config = Configuration().configuration
        self.environment = RandomTargetEnvironment() if self.config.environment.random_target else FixedTargetEnvironment()
        self.optimizer = IQLOptimizer(self.environment)

    def train(self, arguments: argparse.Namespace):
        """Train the brittle star locomotion model.
        This method initializes the training process, including setting up the environment and optimizer, and then runs the optimization loop. 
        After training, it saves the model checkpoint.
        
        :param arguments: Command-line arguments containing configuration and checkpoint information.
        """

        started_at = time.perf_counter()
        self.logger.info("Starting training process...")
        
        self.logger.info(f"Training with {self.environment.number_of_environments} parallel environments")
        self.optimizer.optimize()
        
        checkpoint_base = normalize_checkpoint_base_name(arguments.checkpoint, self.config.checkpoint_directory)
        self.optimizer.save_model(checkpoint_prefix=checkpoint_base)
        
        elapsed = time.perf_counter() - started_at
        self.logger.info(f"Training completed in {elapsed:.1f}s")
        self.logger.debug("Training process finished.")