import jax.numpy as jnp

def compute_z(context, params):
    return jnp.dot(params["global"]["z_W"], context) + params["global"]["z_b"]