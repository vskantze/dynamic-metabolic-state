import numpy as np
import pandas as pd
from glucose_model.data.preprocess import (
    interpolate_meal,
    normalize_context
)
from glucose_model.data.dataset import GlucoseDataset, create_batch
from glucose_model.utils.init_params import init_params
from glucose_model.training.train_step import train_step
from glucose_model.model.model import forward_batch
from glucose_model.data.preprocess import clean_response, clean_sleep, build_meal_trajectories, aggregate_activity, attach_context
from glucose_model.training.loss import loss_fn
import matplotlib.pyplot as plt

import optax

from pathlib import Path
import os 

ROOT_DIR = Path.cwd()
df_activity = pd.read_csv(os.path.join(ROOT_DIR, "data/processed/activity_data.csv"))
print( "Activity data: \n", df_activity.head() )

df_sleep = pd.read_csv(os.path.join(ROOT_DIR, "data/processed/sleep_data.csv"))
print("Sleep data: \n" ,df_sleep.head())

df_response = pd.read_csv(os.path.join(ROOT_DIR, "data/processed/t2d_long_format.csv"))
print("Response data: \n", df_response.head())
print("Response cols: \n", list(df_response.columns))

# Preprocess
df_response = clean_response(df_response)
meals = build_meal_trajectories(df_response)

df_sleep_clean = clean_sleep(df_sleep)
df_activity_agg = aggregate_activity(df_activity)

meals = attach_context(meals, df_sleep_clean, df_activity_agg)

# ---- create time grid ----
t_grid = np.linspace(0, 180, 100)

# ---- interpolate ----
meals_interp = [interpolate_meal(m, t_grid) for m in meals]

# ---- normalize ----
meals_interp, mean, std = normalize_context(meals_interp)

# ---- dataset ----
person_ids = sorted(set(m["person_id"] for m in meals_interp))
person_to_idx = {pid: i for i, pid in enumerate(person_ids)}
for m in meals_interp:
    m["person_idx"] = person_to_idx[m["person_id"]]

dataset = GlucoseDataset(meals_interp)

params = init_params(person_ids)

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

for epoch in range(500):
    indices = np.random.choice(len(dataset), size=8, replace=False)

    batch = create_batch(dataset, indices)

    params, opt_state, loss = train_step(
        params, opt_state, batch
    )

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.3f}")




pred = forward_batch(params, batch)

for idx, _ in enumerate(pred):
    plt.plot(batch["time"][idx], batch["glucose"][idx], c="k", label="true")
    plt.plot(batch["time"][idx], pred[idx],'--r', label="pred")

plt.legend()
plt.title("Model fit")
plt.show()

print(params)