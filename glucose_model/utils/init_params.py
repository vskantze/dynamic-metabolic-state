import jax.numpy as jnp

def init_params(person_ids):

    N_persons = len(person_ids)

    params = {
        "global": {
            "z_W": jnp.ones(5) * 0.1,
            "z_b": 0.0,

            "SI_base": 0.01,
            "alpha": 0.2,

            "SG": 0.01,
            "p3": 0.01,
            "Gb": 0.0,

            "tau_base": 30.0,
            "tau2": 30.0,
            "w": 0.5,
            
        },
        "individual": {
            "SI_base": jnp.ones(N_persons) * 0.01,
            "f": jnp.ones(N_persons) * 0.1,
            "p2": jnp.ones(N_persons) * 0.001,
            "beta_protein": jnp.ones(N_persons) * 0.1,
            "alpha_fat": jnp.ones(N_persons) * 0.1,

        }
        

    }
    return params