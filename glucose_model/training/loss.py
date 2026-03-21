import jax.numpy as jnp
from glucose_model.model.simulator import simulate

def loss_fn(params, batch):
    pred = simulate(params, batch["meal"], batch["context"], batch["time"])
    return jnp.mean((pred - batch["glucose"]) ** 2)