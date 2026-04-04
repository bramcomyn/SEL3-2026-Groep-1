import logging

from brittle_star_locomotion.util.singleton import singleton

@singleton
class Logger:
    """Singleton class to hold the logger."""
    def __init__(self, verbose: bool = False):
        logging_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.debug("Logger initialized with verbose mode: %s", verbose)
        self.logger = logging.getLogger(__name__)
    
    def info(self, message: str):
        """Logs an info message."""
        self.logger.info(message)

    def debug(self, message: str):
        """Logs a debug message."""
        self.logger.debug(message)

    def warning(self, message: str):
        """Logs a warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Logs an error message."""
        self.logger.error(message)
