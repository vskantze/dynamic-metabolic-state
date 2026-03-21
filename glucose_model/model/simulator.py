import diffrax as dfx
import jax.numpy as jnp

from glucose_model.model.dynamics import dynamics
from glucose_model.model.metabolic_state import compute_z
from glucose_model.model.parameters import compute_SI

def simulate(params, meal, context, t_eval):
    z = compute_z(context, params)
    SI = compute_SI(z, params)

    y0 = jnp.array([params["Gb"], 0.0])

    term = dfx.ODETerm(dynamics)

    sol = dfx.diffeqsolve(
        term,
        dfx.Tsit5(),
        t0=0,
        t1=t_eval[-1],
        dt0=1.0,
        y0=y0,
        args=(params, meal, SI),
        saveat=dfx.SaveAt(ts=t_eval),
    )

    return sol.ys[:, 0]