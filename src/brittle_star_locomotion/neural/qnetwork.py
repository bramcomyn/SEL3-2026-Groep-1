import jax
from flax import nnx


class QNetwork(nnx.Module):
    def __init__(self, input_size: int, output_size: int, rngs: nnx.Rngs, hidden_size: int = 32):
        self.mlp = nnx.List([nnx.Linear(input_size, hidden_size, rngs=rngs), nnx.relu, nnx.Linear(hidden_size, output_size, rngs=rngs), nnx.relu])

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.mlp:
            x = layer(x)
        return x
