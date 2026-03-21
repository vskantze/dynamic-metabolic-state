import jax
import optax

from glucose_model.training.loss import loss_fn

@jax.jit
def train_step(params, opt_state, batch, optimizer):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)

    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss