from omegaconf import OmegaConf


def load_config(config_path: str):
    """Loads a YAML configuration file and returns it as a dictionary."""
    try:
        config = OmegaConf.load(config_path)
        return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        raise
