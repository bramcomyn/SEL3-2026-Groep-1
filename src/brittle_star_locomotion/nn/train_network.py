import jax
import jax.numpy as jnp
import optax
from flax import nnx

from brittle_star_locomotion.nn.q_network import QNetwork


@nnx.jit
def train_step(model, optimizer, x, y, rngs) -> jax.Array:
    def loss_fn(model: QNetwork, rngs: nnx.Rngs):
        y_pred = model(x)
        return jnp.mean((y_pred - y) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model, rngs)
    optimizer.update(model, grads)

    return loss


if __name__ == "__main__":
    print("Training Q-Network...")
    model = QNetwork(5, 5, rngs=nnx.Rngs(0), hidden_size=5)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    rngs = nnx.Rngs(0)
    x, y = jnp.ones((5,)), jnp.ones((5,))

    loss = train_step(model, optimizer, x, y, rngs)
    while loss > 1e-6:
        print(f"Loss {loss}")
        loss = train_step(model, optimizer, x, y, rngs)
