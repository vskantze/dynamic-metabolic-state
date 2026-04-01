import jax.numpy as jnp

def Ra(t, meal, params, person_idx):
    carbs = meal["carbs"]
    fat = meal["fat"]

    tau_fast = params["global"]["tau_fast_base"] 
    + jnp.take(params["individual"]["alpha_fat_fast"], person_idx) * fat

    tau_slow = params["global"]["tau_slow_base"]
    + jnp.take(params["individual"]["alpha_fat_slow"], person_idx) * fat

    w = jnp.take(params["individual"]["w"], person_idx)


    t_meal_i = params["global"]["time_meal"] + delta_t_i
    t_pos = jnp.maximum(t, 0)

    fast = (t_pos / tau_fast) * jnp.exp(-t_pos / tau_fast)
    slow = (t_pos / tau_slow) * jnp.exp(-t_pos / tau_slow)

    return carbs * (w * fast + (1 - w) * slow)