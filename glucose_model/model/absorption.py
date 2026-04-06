import jax.numpy as jnp
import jax

def Ra(t, meal, params, person_idx):
    carbs = meal["carbs"]
    fat = meal["fat"]

    tau_fast = params["global"]["tau_fast_base"] \
        + jnp.take(params["individual"]["alpha_fat_fast"], person_idx) * fat

    tau_slow = params["global"]["tau_slow_base"] \
        + jnp.take(params["individual"]["alpha_fat_slow"], person_idx) * fat

    w = jnp.take(params["individual"]["w"], person_idx)

    # timing uncertainty
    shifts = jnp.array([0.0, 20.0, 40.0])
    weights = jax.nn.softmax(params["global"]["timing_logits"])

    def Ra_shift(shift):
        t_pos = jnp.maximum(t - shift, 0)
        fast = (t_pos / tau_fast) * jnp.exp(-t_pos / tau_fast)
        slow = (t_pos / tau_slow) * jnp.exp(-t_pos / tau_slow)
        return carbs * (w * fast + (1 - w) * slow)

    ra = jnp.sum(weights * jnp.array([Ra_shift(s) for s in shifts]))
    return ra