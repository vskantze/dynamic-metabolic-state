import jax.numpy as jnp
from glucose_model.model.model import forward_batch

def loss_fn(params, batch):
    pred = forward_batch(params, batch)
    return jnp.mean((pred - batch["glucose"]) ** 2)