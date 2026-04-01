import pandas as pd
import numpy as np

COLUMN_MAP = {
    "carbs": "CHO_g",
    "fat": "Fat_g",
    "protein": "Prot_g",
}

def clean_response(df):
    # fix dtype warning
    df = df.copy()

    # drop useless columns
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # ensure numeric
    df["Glucose"] = pd.to_numeric(df["Glucose"], errors="coerce")
    df["time_meal"] = pd.to_numeric(df["time_meal"], errors="coerce")

    # drop NaNs
    df = df.dropna(subset=["Glucose", "time_meal", "abs_ID"])

    return df


def build_meal_trajectories(df):
    meals = []

    grouped = df.groupby("abs_ID")

    for meal_id, g in grouped:
        g = g.sort_values("time_meal")
        t = g["time_meal"].values
        t = t - t.min()
        sample = {
            "person_id": int(g["ID"].iloc[0]),
            "meal_id": meal_id,
            "meal_type": g["Meal type"].iloc[0],

            # meal composition (take first row)
            "meal": {
                "carbs": g[COLUMN_MAP["carbs"]].iloc[0],
                "fat": g[COLUMN_MAP["fat"]].iloc[0],
                "protein": g[COLUMN_MAP["protein"]].iloc[0],
            },
            
            "time": t,
            "glucose": g["Glucose"].values,

            # for later merging
            "day": g["Day number"].iloc[0],
        }
        
        # Skip responses not reaching above baseline
        n_below_baseline = sum( (sample["glucose"] - sample["glucose"][0])<0 )
        threshold = 0.8
        if ( n_below_baseline > threshold * len(sample["glucose"]) ):
            continue

        meals.append(sample)

    return meals


def clean_sleep(df_sleep):
    df = df_sleep.copy()

    df = df.rename(columns={
        "Day_Since_First": "day",
        "ID": "person_id"
    })

    # aggregate to ensure uniqueness
    df = df.groupby(["person_id", "day"]).agg({
        "Sleep_Efficiency": "mean",
        "Total_Sleep_Time_TST": "mean"
    }).reset_index()

    return df


def aggregate_activity(df_activity):
    df = df_activity.copy()

    df = df.rename(columns={
        "Day_Since_First": "day",
        "ID": "person_id"
    })

    df = df.groupby(["person_id", "day"]).agg({
        "METs": "mean",
        "kcals": "sum"
    }).reset_index()

    return df

MEAL_TYPE_MAP = {
    "breakfast": [1, 0, 0],
    "lunch": [0, 1, 0],
    "dinner": [0, 0, 1],
}

def attach_context(meals, df_sleep, df_activity):
    sleep_dict = df_sleep.set_index(["person_id", "day"]).to_dict("index")
    act_dict = df_activity.set_index(["person_id", "day"]).to_dict("index")

    enriched = []

    for m in meals:
        key = (m["person_id"], m["day"])

        sleep = sleep_dict.get(key, {})
        act = act_dict.get(key, {})
        meal_type = m.get("meal_type", "unknown")

        meal_vec = MEAL_TYPE_MAP.get(meal_type.lower(), [0, 0, 0])
        context = [
            sleep.get("Sleep_Efficiency", 0.0),
            act.get("METs", 0.0),
            *meal_vec
        ]

        m["context"] = context

        enriched.append(m)

    return enriched

def interpolate_meal(meal, t_grid):
    t = np.array(meal["time"])
    g = np.array(meal["glucose"])

    # sort (safety)
    idx = np.argsort(t)
    t = t[idx]
    g = g[idx]

    # interpolate
    g_interp = np.interp(t_grid, t, g)

    meal_new = meal.copy()
    meal_new["time"] = t_grid
    meal_new["glucose"] = g_interp

    return meal_new

def normalize_context(meals):
    #TODO: Fix the indexing of numeric contexts
    contexts = np.array([m["context"][0:1] for m in meals]) 
    
    mean = contexts.mean(axis=0)
    std = contexts.std(axis=0) + 1e-6

    for m in meals:
        m["context"][0:1] = (np.array(m["context"][0:1]) - mean) / std

    return meals, mean, std