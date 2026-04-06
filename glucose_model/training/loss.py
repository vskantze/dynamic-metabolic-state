import jax.numpy as jnp
from glucose_model.model.model import forward_batch

def loss_fn(params, batch):
    G_pred = forward_batch(params, batch)
    G_obs = batch["glucose"]

    sigma = params["global"]["sigma"]

    residual = G_obs - G_pred

    loss = jnp.mean((residual**2) / (sigma**2) + jnp.log(sigma**2))

    return loss
