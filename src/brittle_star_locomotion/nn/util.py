"""Utility functions for training and evaluating neural networks.
"""
from flax import nnx


def update_model_parameters(model: nnx.Module, copy_from: nnx.Module) -> nnx.Module:
    """Update the parameters of the model by copying them from another model.

    :param nnx.Module model: The model to update.
    :param nnx.Module copy_from: The model to copy parameters from.
    :return: The updated model.
    """
    _, copy_from_state = nnx.split(copy_from)
    nnx.update(model, copy_from_state)

    return model
