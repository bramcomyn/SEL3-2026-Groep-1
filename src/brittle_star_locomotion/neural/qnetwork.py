import jax
import os
import pickle

from collections.abc import Callable
from flax import nnx

CHECKPOINT_DIR = "checkpoints" # TODO: use value from config file

class QNetwork(nnx.Module):
    """
    A multi-layer perceptron (MLP) designed for Q-value estimation in reinforcement learning.
    This module maps environmental states to action-values through a configurable number 
    of hidden layers and rectified linear unit (ReLU) activations.

    :param input_size: The dimensionality of the input state vector.
    :param output_size: The number of discrete actions (size of the output Q-vector).
    :param rngs: A collection of NNX random number generators used for parameter initialization.
    :param hidden_size: The number of neurons in each hidden layer. Defaults to 32.
    :param amount_of_hidden_layers: The number of additional hidden layers to insert 
        between the initial projection and the output layer. Defaults to 0.
    """
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        rngs: nnx.Rngs, 
        hidden_size: int = 32,
        amount_of_hidden_layers: int = 0
    ):
        layers = []

        # initial layer projection
        layers.append(nnx.Linear(input_size, hidden_size, rngs=rngs))
        layers.append(nnx.relu)

        # optional additional hidden layers
        for _ in range(amount_of_hidden_layers):
            layers.append(nnx.Linear(hidden_size, hidden_size, rngs=rngs))
            layers.append(nnx.relu)

        # output layer producing raw Q-values (logits)
        layers.append(nnx.Linear(hidden_size, output_size, rngs=rngs))

        # self.mlp is registered as an nnx.List to track sub-module state
        self.mlp = nnx.List(layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Performs a forward pass of the network to compute Q-values for the given input.

        :param x: A JAX array representing the input state (or a batch of states).
        :return: A JAX array of Q-values with shape (..., output_size).
        """
        for layer in self.mlp:
            x = layer(x)
        return x
    
    def update_model_parameters(self, copy_from: nnx.Module):
        """Update the parameters of the model by copying them from another model.

        :param nnx.Module copy_from: The model to copy parameters from.

        >>> model = QNetwork(5, 5, rngs=nnx.Rngs(0), hidden_size=5)
        >>> copy_from = QNetwork(5, 5, rngs=nnx.Rngs(0), hidden_size=5)
        >>> model.update_model_parameters(copy_from)
        """
        model_graphdef, _ = nnx.split(self)
        copy_from_graphdef, copy_from_state = nnx.split(copy_from)

        assert model_graphdef == copy_from_graphdef, "Model architectures must match to copy parameters."

        nnx.update(self, copy_from_state)
    
    def save_checkpoint(self, name: str):
        """Save a checkpoint of the model.

        :param str name: The name of the checkpoint.

        >>> model = MyModel()
        >>> model.save_checkpoint("my_checkpoint")
        """
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        path_to_checkpoint = os.path.join(CHECKPOINT_DIR, name)

        _, state = nnx.split(self)

        pure_dict_state = nnx.to_pure_dict(state)
        with open(path_to_checkpoint, "wb") as output_file:
            pickle.dump(pure_dict_state, output_file)

    @staticmethod
    def load_checkpoint(model_callback: Callable[[], nnx.Module], name: str) -> nnx.Module:
        """Load a checkpoint of a model.

        :param Callable[[], nnx.Module] model_callback:
            A callback that returns the model to load the checkpoint into.
        :param str name: The name of the checkpoint.
        :return: The model with the loaded checkpoint.

        >>> model = QNetwork.load_checkpoint(lambda: MyModel(), "my_checkpoint")
        """
        path_to_checkpoint = os.path.join(CHECKPOINT_DIR, f"{name}")

        abstract_model = nnx.eval_shape(lambda: model_callback())
        graphdef, abstract_state = nnx.split(abstract_model)

        with open(path_to_checkpoint, "rb") as input_file:
            restored_pure_dict = pickle.load(input_file)

        nnx.replace_by_pure_dict(abstract_state, restored_pure_dict)

        model = nnx.merge(graphdef, abstract_state)
        return model
