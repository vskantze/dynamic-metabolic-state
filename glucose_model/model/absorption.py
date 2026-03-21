import jax.numpy as jnp

def Ra(t, meal, params):
    carbs = meal["carbs"]
    tau = params["tau"]

    return carbs * (t / tau) * jnp.exp(-t / tau)