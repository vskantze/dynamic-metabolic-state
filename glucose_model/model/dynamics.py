import jax.numpy as jnp
from glucose_model.model.absorption import Ra

def dynamics(t, y, args):
    G, X = y
    params, meal, SI, person_idx = args

    f = jnp.take(params["individual"]["f"], person_idx)
    p2 = jnp.take(params["individual"]["p2"], person_idx)
    beta_protein = jnp.take(params["individual"]["beta_protein"], person_idx)

    p3 = params["global"]["p3"]
    Gb = params["global"]["Gb"]
    SG = params["global"]["SG"]
    Ra_t = Ra(t, meal, params, person_idx)

    dGdt = -SG * (G-Gb) - SI * X + f * Ra_t
    dXdt = -p2 * X + p3 * G + beta_protein * meal["protein"]/100

    return jnp.array([dGdt, dXdt])