import jax
from flax import nnx


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
