import jax.numpy as jnp
from glucose_model.model.model import forward_batch

def loss_fn(params, batch, n_samples):
    pred = forward_batch(params, batch, n_samples)
    return jnp.mean((pred - batch["glucose"]) ** 2)