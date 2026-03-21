import jax.numpy as jnp
import matplotlib.pyplot as plt

from glucose_model.utils.init_params import init_params
from glucose_model.model.simulator import simulate

def main():
    params = init_params()

    meal = {"carbs": 60.0}
    context = jnp.array([0.5, 0.3])

    t = jnp.linspace(0, 180, 100)

    glucose = simulate(params, meal, context, t)

    plt.plot(t, glucose)
    plt.xlabel("Time (min)")
    plt.ylabel("Glucose")
    plt.title("Simulated Glucose Response")
    plt.show()

if __name__ == "__main__":
    main()