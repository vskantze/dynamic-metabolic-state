import jax.numpy as jnp

def compute_z(context, params):
    return jnp.dot(params["z_W"], context) + params["z_b"]