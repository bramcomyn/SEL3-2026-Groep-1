import jax
from flax import nnx


class QNetwork(nnx.Module):
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        rngs: nnx.Rngs, 
        hidden_size: int = 32,
        amount_of_hidden_layers: int = 0
    ):
        layers = []

        layers.append(nnx.Linear(input_size, hidden_size, rngs=rngs))
        layers.append(nnx.relu)

        for _ in range(amount_of_hidden_layers):
            layers.append(nnx.Linear(hidden_size, hidden_size, rngs=rngs))
            layers.append(nnx.relu)

        layers.append(nnx.Linear(hidden_size, output_size, rngs=rngs))
        # TODO: no final activation

        self.mlp = nnx.List(layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.mlp:
            x = layer(x)
        return x
