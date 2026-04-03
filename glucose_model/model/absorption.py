import jax.numpy as jnp
import jax.random as random

def Ra(t, meal, params, person_idx, key):
    carbs = meal["carbs"]
    fat = meal["fat"]

    tau_fast = params["global"]["tau_fast_base"] 
    + jnp.take(params["individual"]["alpha_fat_fast"], person_idx) * fat

    tau_slow = params["global"]["tau_slow_base"]
    + jnp.take(params["individual"]["alpha_fat_slow"], person_idx) * fat

    w = jnp.take(params["individual"]["w"], person_idx)


    t_meal_i = random.normal(key, shape=(100,))*params["global"]["time_std"] + params["global"]["time_meal"]
    t_pos = jnp.maximum(t-t_meal_i, 0)

    fast = (t_pos / tau_fast) * jnp.exp(-t_pos / tau_fast)
    slow = (t_pos / tau_slow) * jnp.exp(-t_pos / tau_slow)
    
    ra = jnp.mean( carbs * (w * fast + (1 - w) * slow) )
    return ra, key