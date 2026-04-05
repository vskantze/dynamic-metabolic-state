import numpy as np
import pandas as pd
import yaml
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
from collections import defaultdict

import optax

from pathlib import Path
import os 

# Read data
ROOT_DIR = Path.cwd()
df_activity = pd.read_csv(os.path.join(ROOT_DIR, "data/processed/activity_data.csv"))
print( "Activity data: \n", df_activity.head() )

df_sleep = pd.read_csv(os.path.join(ROOT_DIR, "data/processed/sleep_data.csv"))
print("Sleep data: \n" ,df_sleep.head())

df_response = pd.read_csv(os.path.join(ROOT_DIR, "data/processed/t2d_long_format.csv"))
print("Response data: \n", df_response.head())
print("Response cols: \n", list(df_response.columns))

# Read config
with open(os.path.join(ROOT_DIR, "glucose_model/utils/config.yaml"), "r") as f:
    config = yaml.safe_load(f)
print(config)
n_samples = config["n_samples"]


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

for epoch in range(200):
    indices = np.random.choice(len(dataset), size=200, replace=False)

    batch = create_batch(dataset, indices)

    params, opt_state, loss = train_step(
        params, opt_state, batch, n_samples
    )

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.3f}")




pred = forward_batch(params, batch, n_samples)

fig, ax = plt.subplots(2,1)
residuals = defaultdict(list)

for idx, _ in enumerate(pred):
    if idx == 0:
        ax[0].plot(batch["time"][idx], batch["glucose"][idx], c="k", label="true")
        ax[0].plot(batch["time"][idx], pred[idx],'--r', label="pred")
    else:
        ax[0].plot(batch["time"][idx], batch["glucose"][idx], c="k")
        ax[0].plot(batch["time"][idx], pred[idx],'--r')
    
    res = batch["glucose"][idx] - pred[idx]
    for i, t in enumerate(batch["time"][idx].tolist()):
        key = str(round(t,0))
        residuals[key].append(res[i].item())

df_residuals = pd.DataFrame(residuals)
mean_traj = df_residuals.mean(0)
time_traj = [float(a) for a in mean_traj.index.values]
ax[1].plot(time_traj, mean_traj.values,"--k", label= "Mean")
ax[1].fill_between(time_traj, mean_traj - df_residuals.std(0),
           mean_traj + df_residuals.std(0), label= "Std")
plt.legend()
plt.title("Model fit")
plt.show()

for d, v in params["global"].items():
    print(f"Global param {d}:", v)

for d, v in params["individual"].items():
    print(f"Individual {d}, mean:", v.mean(), "std:", v.std())