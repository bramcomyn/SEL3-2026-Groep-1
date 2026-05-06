import logging

import wandb
from omegaconf import OmegaConf

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

        self._wandb_enabled = False

    def initialize_wandb(self, project: str, config=None, enabled: bool = True) -> None:
        """Initialize a wandb run once for the current process."""
        self._wandb_enabled = enabled
        if not self._wandb_enabled:
            return

        if wandb.run is None:
            wandb_config = self._build_wandb_config(config)
            run_name = self._build_wandb_run_name(config)
            wandb.init(
                entity="comyn-bram-universiteit-gent",
                project=project,
                config=wandb_config,
                name=run_name,
            )

    def _build_wandb_run_name(self, config) -> str:
        """Build a stable wandb run name from the current configuration."""
        if config is None:
            return "brittle-star-locomotion"

        if OmegaConf.is_config(config):
            config = OmegaConf.to_container(config, resolve=True)

        if not isinstance(config, dict):
            return "brittle-star-locomotion"

        rl_config = config.get("rl", {}) or {}
        environment_config = config.get("environment", {}) or {}

        distance_target = environment_config.get("target_distance")
        number_of_episodes = rl_config.get("n_episodes")
        episode_length = environment_config.get("simulation_time")
        damage = environment_config.get("include_damage")

        return (
            f"target-distance_{distance_target}_"
            f"number-of-episodes_{number_of_episodes}_"
            f"episode-length_{episode_length}_"
            f"damage_{damage}"
        )

    def _build_wandb_config(self, config) -> dict:
        """Extract a compact set of hyperparameters for wandb tracking."""
        if config is None:
            return {}

        if OmegaConf.is_config(config):
            config = OmegaConf.to_container(config, resolve=True)

        if not isinstance(config, dict):
            return {}

        rl_config = config.get("rl", {}) or {}
        environment_config = config.get("environment", {}) or {}
        gait_config = config.get("gait", {}) or {}
        cpg_config = config.get("cpg", {}) or {}

        return {
            "seed": rl_config.get("seed"),
            "epsilon": rl_config.get("epsilon"),
            "epsilon_decay": rl_config.get("epsilon_decay"),
            "epsilon_min": rl_config.get("epsilon_min"),
            "learning_rate": rl_config.get("learning_rate"),
            "gamma": rl_config.get("gamma"),
            "hidden_size": rl_config.get("hidden_size"),
            "amount_of_hidden_layers": rl_config.get("amount_of_hidden_layers"),
            "shared_params": rl_config.get("shared_params"),
            "n_episodes": rl_config.get("n_episodes"),
            "batch_size": rl_config.get("batch_size"),
            "target_update_freq": rl_config.get("target_update_freq"),
            "replay_buffer_size": rl_config.get("replay_buffer_size"),
            "number_of_arms": environment_config.get("number_of_arms"),
            "number_of_segments_per_arm": environment_config.get("number_of_segments_per_arm"),
            "number_of_environments": environment_config.get("number_of_environments"),
            "target_distance": environment_config.get("target_distance"),
            "simulation_time": environment_config.get("simulation_time"),
            "number_of_physics_steps_per_control_step": environment_config.get("number_of_physics_steps_per_control_step"),
            "time_scale": environment_config.get("time_scale"),
            "render_every_x_frames": environment_config.get("render_every_x_frames"),
            "fixed_number_of_evaluation_modulation_steps": gait_config.get("fixed_number_of_evaluation_modulation_steps"),
            "fixed_number_of_evaluation_substeps_per_modulation": gait_config.get("fixed_number_of_evaluation_substeps_per_modulation"),
            "cpg_time_step": cpg_config.get("time_step"),
            "cpg_base_frequency_multiplier": cpg_config.get("base_frequency_multiplier"),
            "cpg_coupling_strength": cpg_config.get("coupling_strength"),
        }

    def log_metrics(self, metrics, step: int | None = None) -> None:
        """Log scalar metrics to wandb when enabled."""
        if not self._wandb_enabled:
            return

        wandb.log(metrics, step=step)

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
