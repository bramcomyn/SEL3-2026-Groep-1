from omegaconf import OmegaConf

CONFIG_PATH = "src/brittle_star_locomotion/config/config.yaml"

def load_config(file_path=CONFIG_PATH):
    """Load the configuration from the YAML file."""
    # Load the YAML config
    cfg = OmegaConf.load(file_path)

    return cfg

def load_env_config(file_path=CONFIG_PATH):
    """Load the environment configuration from the YAML file."""
    cfg = OmegaConf.load(file_path)
    return cfg.env

def load_rl_config(file_path=CONFIG_PATH):
    """Load the RL configuration from the YAML file."""
    cfg = OmegaConf.load(file_path)
    return cfg.rl



### Example usage

# Load both RL and environment configurations
# config = load_config()
# rl = load_rl_config()
# env = load_env_config()

# learning_rate = rl.learning_rate
# num_arms = env.num_arms
