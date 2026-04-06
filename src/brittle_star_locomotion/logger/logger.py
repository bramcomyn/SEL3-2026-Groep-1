import logging

from brittle_star_locomotion.util.singleton import Singleton

class Logger:
    """
    Logger class for the brittle star locomotion project.
    """
    def __init__(self, name: str="brittle_star_locomotion", verbose: bool=False):
        """
        Initialize the logger.

        :param name: Name of the logger.
        :param verbose: Whether to print verbose logs.
        """
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
