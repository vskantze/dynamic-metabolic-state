import jax.numpy as jnp
from glucose_model.model.absorption import Ra

def dynamics(t, y, args):
    G, X = y
    params, meal, SI = args

    SG = params["SG"]
    p2 = params["p2"]
    p3 = params["p3"]
    Gb = params["Gb"]

    Ra_t = Ra(t, meal, params)

    dGdt = - (SG + SI * X) * (G - Gb) + Ra_t
    dXdt = -p2 * X + p3 * (G - Gb)

    return jnp.array([dGdt, dXdt])