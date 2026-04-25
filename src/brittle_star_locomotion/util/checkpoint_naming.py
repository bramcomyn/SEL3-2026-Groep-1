import os


def normalize_checkpoint_base_name(checkpoint_argument: str, checkpoint_directory: str) -> str:
    """Convert checkpoint CLI input to a basename used by QNetwork.load_checkpoint."""
    normalized = checkpoint_argument.replace("\\", "/")
    checkpoint_directory = checkpoint_directory.rstrip("/")

    if os.path.isabs(normalized):
        return os.path.basename(normalized)

    if normalized.startswith(f"{checkpoint_directory}/"):
        normalized = normalized[len(checkpoint_directory) + 1 :]

    normalized = normalized.split("/")[-1]

    return normalized

def resolve_agent_checkpoint_name(checkpoint_base: str, agent_id: int, checkpoint_directory: str) -> str:
    """Resolve the checkpoint filename for a specific agent."""
    checkpoint_name = f"{checkpoint_base}_{agent_id}"
    checkpoint_path = os.path.join(checkpoint_directory, checkpoint_name)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint for agent {agent_id} was not found at {checkpoint_path}. "
            f"Expected naming pattern: <checkpoint_prefix>_<agent_id>."
        )

    return checkpoint_name