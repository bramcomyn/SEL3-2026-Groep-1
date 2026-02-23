"""
Checkpointing utilities for saving and loading model checkpoints.
https://flax.readthedocs.io/en/stable/guides/checkpointing.html
"""
import os
import pickle
from collections.abc import Callable

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
    path_to_checkpoint = os.path.join(CHECKPOINT_DIR, name)

    _, state = nnx.split(model)

    pure_dict_state = nnx.to_pure_dict(state)
    with open(path_to_checkpoint, "wb") as output_file:
        pickle.dump(pure_dict_state, output_file)


def load_checkpoint(model_callback: Callable[[], nnx.Module], name: str) -> nnx.Module:
    """Load a checkpoint of a model.

    :param Callable[[], nnx.Module] model_callback:
        A callback that returns the model to load the checkpoint into.
    :param str name: The name of the checkpoint.
    :return: The model with the loaded checkpoint.

    >>> model = load_checkpoint(lambda: MyModel(), "my_checkpoint")
    """
    path_to_checkpoint = os.path.join(CHECKPOINT_DIR, f"{name}")

    abstract_model = nnx.eval_shape(lambda: model_callback())
    graphdef, abstract_state = nnx.split(abstract_model)

    with open(path_to_checkpoint, "rb") as input_file:
        restored_pure_dict = pickle.load(input_file)

    nnx.replace_by_pure_dict(abstract_state, restored_pure_dict)

    model = nnx.merge(graphdef, abstract_state)
    return model
