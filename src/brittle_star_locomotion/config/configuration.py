from omegaconf import OmegaConf

from brittle_star_locomotion.util.singleton import Singleton

class Configuration(metaclass=Singleton):
    """
    Configuration class for the brittle star locomotion project.
    """
    def __init__(self, configuration_file_path: str="configs/base_config.yaml"):
        """
        Initialize the configuration.

        :param configuration_file_path: Path to the configuration file.
        """
        self.configuration_file_path = configuration_file_path
        self._load_configuration()
        
    def _load_configuration(self):
        """
        Load the configuration from the file.
        """
        try:
            self.configuration = OmegaConf.load(self.configuration_file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from {self.configuration_file_path}: {e}")
