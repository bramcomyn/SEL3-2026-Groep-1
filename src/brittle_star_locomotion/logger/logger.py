import logging

from brittle_star_locomotion.util.singleton import Singleton


class _ColorFormatter(logging.Formatter):
    """Formatter that colors log levels for terminal output."""

    COLORS = {
        logging.DEBUG: "\033[36m",    # Cyan
        logging.INFO: "\033[32m",     # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",    # Red
        logging.CRITICAL: "\033[35m", # Magenta
    }

    NAME_COLOR = "\033[34m"  # Blue
    TIME_COLOR = "\033[90m"  # Bright black / gray
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)

        original_levelname = record.levelname
        original_name = record.name
        original_asctime = getattr(record, "asctime", None)

        record.levelname = f"{color}{record.levelname}{self.RESET}"
        record.name = f"{self.NAME_COLOR}{record.name}{self.RESET}"

        try:
            record.message = record.getMessage()
            if self.usesTime():
                record.asctime = f"{self.TIME_COLOR}{self.formatTime(record, self.datefmt)}{self.RESET}"
            return self.formatMessage(record)
        finally:
            record.levelname = original_levelname
            record.name = original_name

            if original_asctime is None:
                delattr(record, "asctime")
            else:
                record.asctime = original_asctime

class Logger(metaclass=Singleton):
    """
    Logger class for the brittle star locomotion project.
    """
    def __init__(self, name: str="brittle_star_locomotion", verbose: bool=False):
        """
        Initialize the logger.

        :param name: Name of the logger.
        :param verbose: Whether to print verbose logs.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                _ColorFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )

            self.logger.addHandler(handler)
            self.logger.propagate = False

    def debug(self, message: str):
        """Log a debug message to the console."""
        self.logger.debug(message)

    def info(self, message: str):
        """Log an info message to the console."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log a warning message to the console."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log an error message to the console."""
        self.logger.error(message)
