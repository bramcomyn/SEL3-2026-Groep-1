"""Utility functions for training and evaluating neural networks.
"""
from flax import nnx


def update_model_parameters(model: nnx.Module, copy_from: nnx.Module) -> nnx.Module:
    """Update the parameters of the model by copying them from another model.

    :param nnx.Module model: The model to update.
    :param nnx.Module copy_from: The model to copy parameters from.
    :return: The updated model.

    >>> model = QNetwork(5, 5, rngs=nnx.Rngs(0), hidden_size=5)
    >>> copy_from = QNetwork(5, 5, rngs=nnx.Rngs(0), hidden_size=5)
    >>> updated_model = update_model_parameters(model, copy_from)
    """
    model_graphdef, _ = nnx.split(model)
    copy_from_graphdef, copy_from_state = nnx.split(copy_from)

    assert model_graphdef == copy_from_graphdef, \
        "Model architectures must match to copy parameters."

    nnx.update(model, copy_from_state)
    return model
