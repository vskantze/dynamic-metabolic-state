"""Synthetic A/T prior-learning diagnostic for the live Mode 2 notebook.

The goal is intentionally narrow: make a synthetic person whose true meal
amplitude/duration distribution is broader than the initial prior, then show
whether sequential meal labels shift and widen the prior toward that person's
distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


A_BOUNDS = (0.01, 0.35)
T_BOUNDS = (15.0, 180.0)


@dataclass(frozen=True)
class BroadATDiagnosticConfig:
    n_meals: int = 240
    random_state: int = 13
    prior_A: float = 0.055
    prior_T: float = 55.0
    prior_log_A_sd: float = 0.18
    prior_log_T_sd: float = 0.16
    prior_strength: float = 10.0
    label_log_sd: float = 0.08
    snapshots: tuple[int, ...] = (0, 5, 15, 50, 150, 240)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _simulate_meal_nutrients(rng: np.random.Generator, meal_type: str) -> dict[str, float]:
    """Small local copy so the diagnostic is independent of notebook cells."""

    if meal_type == "breakfast":
        carbs = rng.lognormal(np.log(55.0), 0.45)
        fat = rng.lognormal(np.log(15.0), 0.50)
        protein = rng.lognormal(np.log(20.0), 0.38)
    elif meal_type == "dinner":
        carbs = rng.lognormal(np.log(82.0), 0.48)
        fat = rng.lognormal(np.log(31.0), 0.48)
        protein = rng.lognormal(np.log(36.0), 0.36)
    else:
        carbs = rng.lognormal(np.log(72.0), 0.45)
        fat = rng.lognormal(np.log(24.0), 0.48)
        protein = rng.lognormal(np.log(30.0), 0.35)

    carbs = float(np.clip(carbs, 12.0, 190.0))
    fat = float(np.clip(fat, 2.0, 120.0))
    protein = float(np.clip(protein, 4.0, 120.0))
    fiber = float(np.clip(rng.normal(7.0 + 0.05 * carbs, 4.0), 0.0, 35.0))
    sugars = float(np.clip(rng.normal(0.30 * carbs, 0.16 * carbs + 4.0), 0.0, carbs))
    glycemic_index = float(np.clip(rng.normal(55.0, 12.0), 25.0, 95.0))
    glycemic_load = float(carbs * glycemic_index / 100.0)
    kcal = float(4.0 * carbs + 9.0 * fat + 4.0 * protein)

    return {
        "carbs": carbs,
        "fat": fat,
        "protein": protein,
        "fiber": fiber,
        "sugars": sugars,
        "glycemic_index": glycemic_index,
        "glycemic_load": glycemic_load,
        "kcal": kcal,
    }


def simulate_broad_person_at_distribution(
    n_meals: int = 240,
    random_state: int = 13,
) -> pd.DataFrame:
    """Simulate one person with broad, meal-dependent true A/T values."""

    rng = np.random.default_rng(int(random_state))
    meal_types = np.asarray(["breakfast", "lunch", "dinner"], dtype=object)
    meal_probs = np.asarray([0.30, 0.38, 0.32], dtype=float)

    subtype_effects = {
        "usual": (0.00, 0.00),
        "fast_high": (0.55, -0.40),
        "slow_low": (-0.35, 0.45),
        "slow_high": (0.40, 0.55),
        "small_fast": (-0.55, -0.50),
    }
    subtype_names = np.asarray(list(subtype_effects), dtype=object)
    subtype_probs = np.asarray([0.38, 0.20, 0.18, 0.16, 0.08], dtype=float)

    person_log_A = rng.normal(0.0, 0.20)
    person_log_T = rng.normal(0.0, 0.22)
    cat_A = {meal_type: rng.normal(0.0, 0.22) for meal_type in meal_types}
    cat_T = {meal_type: rng.normal(0.0, 0.28) for meal_type in meal_types}

    rows: list[dict[str, Any]] = []
    for meal_order in range(int(n_meals)):
        meal_type = str(rng.choice(meal_types, p=meal_probs))
        meal = _simulate_meal_nutrients(rng, meal_type)
        subtype = str(rng.choice(subtype_names, p=subtype_probs))
        subtype_A, subtype_T = subtype_effects[subtype]

        carbs = _finite_float(meal["carbs"])
        fat = _finite_float(meal["fat"])
        protein = _finite_float(meal["protein"])
        fiber = _finite_float(meal["fiber"])
        sugars = _finite_float(meal["sugars"])
        glycemic_load = _finite_float(meal["glycemic_load"])
        kcal = _finite_float(meal["kcal"])

        z_carbs = (carbs - 70.0) / 45.0
        z_fat = (fat - 24.0) / 18.0
        z_protein = (protein - 28.0) / 18.0
        z_gl = (glycemic_load - 38.0) / 28.0
        z_fiber = (fiber - 7.0) / 5.0
        z_kcal = (kcal - 600.0) / 350.0

        log_A = (
            np.log(0.115)
            + person_log_A
            + cat_A[meal_type]
            + subtype_A
            + {"breakfast": 0.18, "lunch": 0.02, "dinner": -0.08}[meal_type]
            + 0.45 * z_carbs
            + 0.20 * z_gl
            + 0.08 * ((sugars - 18.0) / 14.0)
            - 0.12 * z_fat
            + rng.normal(0.0, 0.24)
        )
        log_T = (
            np.log(88.0)
            + person_log_T
            + cat_T[meal_type]
            + subtype_T
            + {"breakfast": -0.20, "lunch": 0.02, "dinner": 0.24}[meal_type]
            + 0.45 * z_fat
            + 0.18 * z_protein
            + 0.16 * z_kcal
            - 0.14 * z_fiber
            + rng.normal(0.0, 0.28)
        )

        rows.append(
            {
                "meal_order": meal_order,
                "meal_type": meal_type,
                "subtype": subtype,
                "A_true": float(np.clip(np.exp(log_A), *A_BOUNDS)),
                "T_true": float(np.clip(np.exp(log_T), *T_BOUNDS)),
                "carbs": carbs,
                "fat": fat,
                "protein": protein,
                "fiber": fiber,
                "glycemic_load": glycemic_load,
                "kcal": kcal,
            }
        )

    return pd.DataFrame(rows)


def init_running_log_at_prior(
    prior_A: float,
    prior_T: float,
    prior_log_A_sd: float,
    prior_log_T_sd: float,
    prior_strength: float,
) -> dict[str, float]:
    n = float(max(prior_strength, 2.0))
    return {
        "n": n,
        "log_A_mean": float(np.log(prior_A)),
        "log_T_mean": float(np.log(prior_T)),
        "log_A_M2": float(n * prior_log_A_sd**2),
        "log_T_M2": float(n * prior_log_T_sd**2),
    }


def running_log_prior_summary(
    state: dict[str, float],
    min_log_sd: float = 0.05,
    max_log_sd: float = 1.25,
) -> dict[str, float]:
    n = float(max(state["n"], 2.0))
    log_A_sd = float(np.sqrt(max(state["log_A_M2"] / n, min_log_sd**2)))
    log_T_sd = float(np.sqrt(max(state["log_T_M2"] / n, min_log_sd**2)))
    log_A_sd = float(np.clip(log_A_sd, min_log_sd, max_log_sd))
    log_T_sd = float(np.clip(log_T_sd, min_log_sd, max_log_sd))

    return {
        "n_seen_effective": n,
        "log_A_mean": float(state["log_A_mean"]),
        "log_T_mean": float(state["log_T_mean"]),
        "log_A_sd": log_A_sd,
        "log_T_sd": log_T_sd,
        "A_mean": float(np.exp(state["log_A_mean"])),
        "T_mean": float(np.exp(state["log_T_mean"])),
    }


def _weighted_running_update(
    n: float,
    mean: float,
    m2: float,
    value: float,
    weight: float = 1.0,
) -> tuple[float, float, float]:
    weight = float(max(weight, 1e-9))
    new_n = n + weight
    delta = value - mean
    new_mean = mean + weight * delta / new_n
    new_m2 = m2 + weight * delta * (value - new_mean)
    return new_n, new_mean, float(max(new_m2, 1e-12))


def update_running_log_at_prior(
    state: dict[str, float],
    A_obs: float,
    T_obs: float,
    weight: float = 1.0,
) -> dict[str, float]:
    log_A = float(np.log(np.clip(A_obs, *A_BOUNDS)))
    log_T = float(np.log(np.clip(T_obs, *T_BOUNDS)))

    n, mean_A, m2_A = _weighted_running_update(
        state["n"],
        state["log_A_mean"],
        state["log_A_M2"],
        log_A,
        weight=weight,
    )
    _, mean_T, m2_T = _weighted_running_update(
        state["n"],
        state["log_T_mean"],
        state["log_T_M2"],
        log_T,
        weight=weight,
    )

    state.update(
        {
            "n": n,
            "log_A_mean": mean_A,
            "log_T_mean": mean_T,
            "log_A_M2": m2_A,
            "log_T_M2": m2_T,
        }
    )
    return state


def prior_truth_coverage(
    truth_df: pd.DataFrame,
    prior_summary: dict[str, float],
) -> dict[str, float]:
    log_A = np.log(np.clip(truth_df["A_true"].to_numpy(dtype=float), 1e-12, None))
    log_T = np.log(np.clip(truth_df["T_true"].to_numpy(dtype=float), 1e-12, None))
    z2 = (
        ((log_A - prior_summary["log_A_mean"]) / max(prior_summary["log_A_sd"], 1e-9)) ** 2
        + ((log_T - prior_summary["log_T_mean"]) / max(prior_summary["log_T_sd"], 1e-9)) ** 2
    )

    return {
        "coverage_50": float(np.mean(z2 <= 1.38629436112)),
        "coverage_90": float(np.mean(z2 <= 4.60517018599)),
        "median_log_density": float(
            np.median(
                -0.5 * z2
                - np.log(
                    2.0
                    * np.pi
                    * max(prior_summary["log_A_sd"], 1e-9)
                    * max(prior_summary["log_T_sd"], 1e-9)
                )
            )
        ),
    }


def run_broad_at_prior_shift_diagnostic(
    config: BroadATDiagnosticConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    if config is None:
        config = BroadATDiagnosticConfig()
    elif isinstance(config, dict):
        config = BroadATDiagnosticConfig(**config)

    rng = np.random.default_rng(int(config.random_state) + 1009)
    truth = simulate_broad_person_at_distribution(
        n_meals=int(config.n_meals),
        random_state=int(config.random_state),
    )
    state = init_running_log_at_prior(
        prior_A=float(config.prior_A),
        prior_T=float(config.prior_T),
        prior_log_A_sd=float(config.prior_log_A_sd),
        prior_log_T_sd=float(config.prior_log_T_sd),
        prior_strength=float(config.prior_strength),
    )

    history: list[dict[str, float]] = []
    for n_seen, row in enumerate(truth.itertuples(index=False)):
        prior = running_log_prior_summary(state)
        history.append(
            {
                "n_seen": int(n_seen),
                **prior,
                **prior_truth_coverage(truth, prior),
            }
        )

        label_A = float(row.A_true) * float(np.exp(rng.normal(0.0, config.label_log_sd)))
        label_T = float(row.T_true) * float(np.exp(rng.normal(0.0, config.label_log_sd)))
        update_running_log_at_prior(state, label_A, label_T)

    prior = running_log_prior_summary(state)
    history.append(
        {
            "n_seen": int(len(truth)),
            **prior,
            **prior_truth_coverage(truth, prior),
        }
    )
    return {
        "truth": truth,
        "history": pd.DataFrame(history),
        "config": config,
    }


def plot_broad_at_prior_shift_diagnostic(result: dict[str, Any]):
    truth = result["truth"].copy()
    history = result["history"].copy()
    config = result["config"]
    max_seen = int(history["n_seen"].max())
    snapshots = [n for n in config.snapshots if n <= max_seen]
    if max_seen not in snapshots:
        snapshots.append(max_seen)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(snapshots)))
    for meal_type, group in truth.groupby("meal_type"):
        ax.scatter(
            group["A_true"],
            group["T_true"],
            s=22,
            alpha=0.28,
            label=f"true {meal_type}",
        )

    A_grid = np.linspace(A_BOUNDS[0], A_BOUNDS[1], 140)
    T_grid = np.linspace(T_BOUNDS[0], T_BOUNDS[1], 140)
    AA, TT = np.meshgrid(A_grid, T_grid)
    for color, n_seen in zip(colors, snapshots):
        row = history.loc[(history["n_seen"] - n_seen).abs().idxmin()]
        z2 = (
            ((np.log(AA) - row["log_A_mean"]) / row["log_A_sd"]) ** 2
            + ((np.log(TT) - row["log_T_mean"]) / row["log_T_sd"]) ** 2
        )
        ax.contour(AA, TT, z2, levels=[4.60517018599], colors=[color], linewidths=2.0)
        ax.scatter(
            row["A_mean"],
            row["T_mean"],
            color=color,
            s=45,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.text(
            row["A_mean"],
            row["T_mean"],
            f" {int(row['n_seen'])}",
            color=color,
            fontsize=9,
            weight="bold",
        )

    ax.set_title("True broad A/T landscape and learned 90% prior contour")
    ax.set_xlabel("A response amplitude")
    ax.set_ylabel("T response duration, min")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[0, 1]
    ax.plot(history["n_seen"], history["coverage_50"], label="50% prior region", color="tab:blue")
    ax.plot(history["n_seen"], history["coverage_90"], label="90% prior region", color="tab:orange")
    ax.axhline(0.50, color="tab:blue", linestyle=":", alpha=0.7)
    ax.axhline(0.90, color="tab:orange", linestyle=":", alpha=0.7)
    ax.set_ylim(0, 1.02)
    ax.set_title("Coverage of the true person-specific A/T distribution")
    ax.set_xlabel("completed meals observed")
    ax.set_ylabel("fraction covered")
    ax.grid(alpha=0.2)
    ax.legend()

    for ax, target, label in [
        (axes[1, 0], "A", "A response amplitude"),
        (axes[1, 1], "T", "T response duration, min"),
    ]:
        true_vals = truth[f"{target}_true"].to_numpy(dtype=float)
        q10, q50, q90 = np.quantile(true_vals, [0.10, 0.50, 0.90])
        x = history["n_seen"].to_numpy(dtype=float)
        log_mean = history[f"log_{target}_mean"].to_numpy(dtype=float)
        log_sd = history[f"log_{target}_sd"].to_numpy(dtype=float)
        mean = history[f"{target}_mean"].to_numpy(dtype=float)
        low = np.exp(log_mean - 1.645 * log_sd)
        high = np.exp(log_mean + 1.645 * log_sd)

        ax.axhspan(q10, q90, color="0.85", alpha=0.65, label="true 10-90%")
        ax.axhline(q50, color="black", linestyle="--", linewidth=1.0, label="true median")
        ax.plot(x, mean, color="tab:green", linewidth=2.0, label="learned prior mean")
        ax.fill_between(x, low, high, color="tab:green", alpha=0.20, label="learned 90% interval")
        ax.set_title(f"Prior shift/expansion for {target}")
        ax.set_xlabel("completed meals observed")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    plt.tight_layout()
    return fig, axes


def broad_at_snapshot_table(result: dict[str, Any]) -> pd.DataFrame:
    history = result["history"]
    config = result["config"]
    rows = [n for n in config.snapshots if n < len(history)]
    rows = list(dict.fromkeys(rows + [len(history) - 1]))
    columns = [
        "n_seen",
        "A_mean",
        "T_mean",
        "log_A_sd",
        "log_T_sd",
        "coverage_50",
        "coverage_90",
        "median_log_density",
    ]
    return history.iloc[rows][columns].reset_index(drop=True)
