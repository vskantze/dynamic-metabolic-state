import jax.numpy as jnp

def Ra(t, meal, params, alpha_fat):
    carbs = meal["carbs"]
    fat = meal["fat"]

    tau = params["global"]["tau_base"] + alpha_fat * fat

    t_pos = jnp.maximum(t-30.0, 0)

    Ra = (t_pos / tau) * jnp.exp(-t_pos / tau)

    return carbs * Ra