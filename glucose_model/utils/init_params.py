import jax.numpy as jnp

def init_params():
    return {
        "z_W": jnp.array([0.1, 0.1]),
        "z_b": 0.0,

        "SI_base": 0.01,
        "alpha": 0.2,

        "SG": 0.01,
        "p2": 0.02,
        "p3": 0.01,
        "Gb": 90.0,

        "tau": 30.0,
    }