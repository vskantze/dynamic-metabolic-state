import jax.numpy as jnp
import jax
import jax.random as jr

def init_params(person_ids):
    N = len(person_ids)
    key = jr.PRNGKey(0)
    
    k1, k2 = jax.random.split(key)

    params = {
        "global": {
            "z_W": jnp.ones(5) * 0.2,
            "z_b": 0.0,

            "SI_base": 0.02,
            "alpha": 0.2,

            "SG": 0.03,
            "p3": 0.03,
            "Gb": 0.0,

            "tau_fast_base": 20.0,
            "tau_slow_base": 80.0,
            "w": 0.6,

            "time_meal": 30.0,
            "time_std": 5.0,
        },

        "individual": {
            "SI_base": 0.02 + 0.005 * jax.random.normal(k1, (N,)),
            "f": jnp.ones(N) * 0.3,
            "p2": jnp.ones(N) * 0.05,
            "beta_protein": jnp.ones(N) * 0.03,

            "alpha_fat_fast": jnp.ones(N) * 0.5,
            "alpha_fat_slow": jnp.ones(N) * 1.0,

            "w": jnp.ones(N) * 0.6,
        }
    }

    return params