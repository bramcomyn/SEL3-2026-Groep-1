import logging

from omegaconf import OmegaConf

from brittle_star_locomotion.util.singleton import singleton


@singleton
class Configuration:
    """Singleton class to hold the configuration."""
    def __init__(self, config_path: str):
        try:
            logging.debug(f"Loading configuration from {config_path}")
            self.config = OmegaConf.load(config_path)
        except Exception as e:
            logging.error(f"Error initializing Configuration: {e}")
            raise

    def get(self, key, default=None):
        """Get a configuration value by key."""
        return self.config.get(key, default)
