import jax
import jax.numpy as jnp
import optax
from flax import nnx
from moojoco.environment.dual import DualMuJoCoEnvironment

from brittle_star_locomotion.nn.q_network import QNetwork


class IQL:
    def __init__(
        self,
        model: QNetwork,
        optimizer: nnx.Optimizer,
        n_agents: int,
        env: DualMuJoCoEnvironment,  # TODO will need wrapper env that modulates CPG
    ):
        self.model = model
        self.optimizer = optimizer
        self.n_agents = n_agents
        self.env = env

        self.value_networks = [
            QNetwork(5, 5, rngs=nnx.Rngs(0), hidden_size=5) for _ in range(n_agents)
        ]
        self.target_networks = [
            QNetwork(5, 5, rngs=nnx.Rngs(0), hidden_size=5) for _ in range(n_agents)
        ]
        self.replay_buffers = [  # TODO: more performant replay buffer
            [] for _ in range(n_agents)
        ]

    def train(self, **kwargs):
        epochs = kwargs.get("epochs", 100)

        for epoch in range(epochs):
            # Collect experience:
            # TODO - Observe current env for each agent (env.observe(agent))
            # TODO - Joint greedy action (epsilon-greedy)
            # TODO - Apply joint action and collect (
            #        env.step(action) = reward, next obs)

            for agent in range(self.n_agents):
                # Store transition in replay buffer
                # Sample mini-batch from replay buffer
                # Compute target Q-values using target network
                # Compute loss
                # Call optimizer
                # Update target network param (interval)
                pass


@nnx.jit
def train_step(model, optimizer, x, y) -> jax.Array:
    def loss_fn(model: QNetwork):
        y_pred = model(x)
        return jnp.mean((y_pred - y) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)

    return loss


if __name__ == "__main__":
    print("Training Q-Network...")
    model = QNetwork(5, 5, rngs=nnx.Rngs(0), hidden_size=5)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    rngs = nnx.Rngs(0)
    x, y = jnp.ones((5,)), jnp.ones((5,))

    loss = train_step(model, optimizer, x, y)
    while loss > 1e-6:
        print(f"Loss {loss}")
        loss = train_step(model, optimizer, x, y)
