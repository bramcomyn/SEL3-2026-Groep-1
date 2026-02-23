"""
Checkpointing utilities for saving and loading model checkpoints.
https://flax.readthedocs.io/en/stable/guides/checkpointing.html
"""
import os
from collections.abc import Callable

import orbax.checkpoint as ocp
from flax import nnx

CHECKPOINT_DIR = "checkpoints"


def save_checkpoint(model: nnx.Module, name: str):
    """Save a checkpoint of the model.

    :param nnx.Module model: The model to checkpoint.
    :param str name: The name of the checkpoint.

    >>> model = MyModel()
    >>> save_checkpoint(model, "my_checkpoint")
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    _, state = nnx.split(model)

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(
        os.path.join(CHECKPOINT_DIR, name),
        state
    )


def load_checkpoint(model_callback: Callable[[], nnx.Module], name: str) -> nnx.Module:
    """Load a checkpoint of a model.

    :param Callable[[], nnx.Module] model_callback:
        A callback that returns the model to load the checkpoint into.
    :param str name: The name of the checkpoint.
    :return: The model with the loaded checkpoint.

    >>> model = load_checkpoint(lambda: MyModel(), "my_checkpoint")
    """
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{name}")

    abstract_model = nnx.eval_shape(lambda: model_callback())
    graphdef, abstract_state = nnx.split(abstract_model)

    checkpointer = ocp.StandardCheckpointer()
    state_restored = checkpointer.restore(checkpoint_path, abstract_state)

    model = nnx.merge(graphdef, state_restored)
    return model
