import diffrax as dfx
import jax.numpy as jnp

from glucose_model.model.dynamics import dynamics
from glucose_model.model.metabolic_state import compute_z

def simulate(params, meal, context, t_eval, person_idx):
    
    SI_base = params["individual"]["SI_base"][person_idx]
    z = compute_z(context, params)
    SI = SI_base * jnp.exp(z)


    y0 = jnp.array([params["global"]["Gb"], 0.0])

    term = dfx.ODETerm(dynamics)

    sol = dfx.diffeqsolve(
        term,
        dfx.Tsit5(),
        t0=0,
        t1=t_eval[-1],
        dt0=1.0,
        y0=y0,
        args=(params, meal, SI, person_idx),
        saveat=dfx.SaveAt(ts=t_eval),
    )

    return sol.ys[:, 0]