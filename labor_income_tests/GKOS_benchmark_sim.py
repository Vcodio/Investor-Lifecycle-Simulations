"""
THIS SCRIPT IS INTENDED TO BE A STAND ALONE TEST FOR GKOS LIFECYCLE EARNINGS MODEL

``typical_worker=True`` matches the base case in Anarkulova, Cederburg,
O'Doherty (2023): median type (alpha, beta, z0) = (0, 0, 0), with stochastic
eta, eps, nu only; they scale simulated levels to GKOS average log earnings
(2010 USD) then CPI. Use ``additional_log_intercept`` for that level shift.

Reference: Guvenen, Karahan, Ozkan, Song, "What Do Data on Millions of U.S.
Workers Reveal about Life-Cycle Earnings Dynamics?"
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

import numpy as np

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Single source of truth with ``LIFECYCLE MODEL/earnings.py`` (lifecycle simulations).
_gbm_path = Path(__file__).resolve().parent.parent / "LIFECYCLE MODEL" / "gkos_benchmark.py"
_spec = importlib.util.spec_from_file_location("gkos_benchmark", _gbm_path)
_gbm = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
# Required so @dataclass can resolve __module__ during exec_module.
sys.modules.setdefault("gkos_benchmark", _gbm)
_spec.loader.exec_module(_gbm)

GKOSBenchmarkParams = _gbm.GKOSBenchmarkParams
simulate_gkos = _gbm.simulate_gkos


def fraction_full_year_nonemployed(y: np.ndarray) -> float:
    """Share of (worker, year) cells with zero earnings."""
    return float(np.mean(y <= 0.0))


def _percentiles_positive_earnings(
    y: np.ndarray, percentiles: tuple[float, ...], min_count: int = 100
) -> tuple[np.ndarray, ...]:
    """
    Per-age percentiles using only strictly positive earnings.

    Raw percentiles include Y=0; whenever Pr(Y=0) >= 10%, the overall P10 is
    exactly zero even though the 10th percentile *among earners* is positive.
    Lifecycle figures in applications often condition on positive income.
    """
    n_t = y.shape[1]
    out = [np.full(n_t, np.nan, dtype=float) for _ in percentiles]
    for t in range(n_t):
        pos = y[:, t][(y[:, t] > 0.0) & np.isfinite(y[:, t])]
        if pos.size < min_count:
            continue
        for k, q in enumerate(percentiles):
            out[k][t] = float(np.percentile(pos, q))
    return tuple(out)


def _fmt_usd_axis(value: float, _pos: int) -> str:
    """Y-axis tick labels with dollar sign and thousands separators."""
    if not np.isfinite(value) or abs(value) >= 1e12:
        return ""
    try:
        return f"${value:,.0f}"
    except (OverflowError, ValueError):
        return ""


def plot_levels(y: np.ndarray, ages: np.ndarray, out_name: str = "gkos_levels.png") -> str:
    plt.style.use("dark_background")
    p10, p25, p50, p75, p90 = _percentiles_positive_earnings(y, (10, 25, 50, 75, 90))

    fig, ax = plt.subplots(
        figsize=(10, 5.2),
        facecolor="#0b0f14",
        constrained_layout=True,
    )
    ax.set_facecolor("#0b0f14")

    # Warm greens + cream median (distinct from prior cyan/blue)
    c_outer = "#3d8f6e"
    c_iqr = "#7ecfae"
    c_median = "#fff4e0"

    ax.fill_between(
        ages,
        p10,
        p90,
        alpha=0.26,
        color=c_outer,
        linewidth=0,
        label="P10-P90",
    )
    ax.fill_between(
        ages,
        p25,
        p75,
        alpha=0.42,
        color=c_iqr,
        linewidth=0,
        label="P25-P75",
    )
    ax.plot(ages, p50, lw=2.4, color=c_median, label="Median")

    ax.set_xlabel("Age", color="#c8d0da", fontsize=11)
    ax.set_ylabel("Earnings (USD)", color="#c8d0da", fontsize=11)
    ax.set_title("Earnings levels", color="#e8eef4", fontsize=13)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_usd_axis))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.tick_params(axis="both", colors="#9aa5b1", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#2a3340")
    ax.grid(True, alpha=0.22, color="#3d4a5c", linestyle="-", linewidth=0.6)
    leg = ax.legend(frameon=True, facecolor="#151b24", edgecolor="#2a3340", fontsize=10)
    for text in leg.get_texts():
        text.set_color("#dce4ee")

    path = os.path.join(PLOT_DIR, out_name)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


if __name__ == "__main__":
    n = 200_000
    # Tuned so P90 among Y>0 at age 47 ~ $160k (bisection, n=150k, seed=1).
    demo_shift = 7.427
    params = GKOSBenchmarkParams(
        additional_log_intercept=demo_shift,
        typical_worker=True,
    )
    logy, y, ages = simulate_gkos(params=params, n_workers=n, seed=0)
    fe = fraction_full_year_nonemployed(y)
    idx47 = int(np.where(ages == 47.0)[0][0])
    pos47 = y[y[:, idx47] > 0.0, idx47]
    p90_47 = float(np.percentile(pos47, 90))
    print("GKOS benchmark (Table IV col. 8), ages 25-60")
    print("Base case: typical_worker=True  (alpha=beta=z0=0), per Cederburg et al. (2023)")
    print(f"Workers x years: {n} x {ages.size}")
    print(f"additional_log_intercept (level scale): {demo_shift}")
    print(f"Fraction zero earnings (full-year nonemployment proxy): {fe:.4f}")
    print(f"P90 earnings at age 47 (among Y>0): ${p90_47:,.0f}")
    idx25, idx55 = 0, 30
    m25 = float(np.median(y[y[:, idx25] > 0, idx25]))
    m55 = float(np.median(y[y[:, idx55] > 0, idx55]))
    print(f"Median earnings age 25 (Y>0): {m25:,.0f}")
    print(f"Median earnings age 55 (Y>0): {m55:,.0f}")
    p = plot_levels(y, ages)
    print(f"Saved plot: {p}")
