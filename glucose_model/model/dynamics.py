import jax.numpy as jnp
from glucose_model.model.absorption import Ra

def dynamics(t, y, args):
    G, X = y
    params, meal, SI, f, p2, beta_protein, alpha_fat = args

    p3 = params["global"]["p3"]
    Gb = params["global"]["Gb"]
    SG = params["global"]["SG"]
    Ra_t = Ra(t, meal, params, alpha_fat)

    dGdt = -SG * (G-Gb) - SI * X + f * Ra_t
    dXdt = -p2 * X + p3 * G + beta_protein * meal["protein"]

    return jnp.array([dGdt, dXdt])