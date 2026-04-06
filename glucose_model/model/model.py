import jax
from glucose_model.model.simulator import simulate

def forward_single(params, batch):
    return simulate(
        params,
        batch["meal"],
        batch["context"],
        batch["time"],
        batch["person_idx"],
    )

forward_batch = jax.vmap(forward_single, in_axes=(None, 0))