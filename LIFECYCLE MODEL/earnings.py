"""
Labor earnings for the lifecycle model: GKOS benchmark (SR 710, Table IV model 8).

Implementation lives in ``gkos_benchmark.py`` (same process as
``Labor Income Tests/GKOS_benchmark_sim.py``). This module scales simulated
levels so median first-year earnings match ``annual_income_real`` via
``baseline_for_median_at_start``.

Configurable via ``config.gkos_params``:
  - ``GKOS_TYPICAL_WORKER`` (bool, default True): median type (alpha, beta, z0) = 0.
  - ``GKOS_ADDITIONAL_LOG_INTERCEPT`` (float, default 7.427): level shift in the exponent.

Legacy keys (ETA_*, NU_SCAR_*, AGE_PEAK, etc.) are ignored by the benchmark engine
but may remain in the dict for Streamlit compatibility.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np


def _load_gkos_benchmark():
    """
    Load sibling ``gkos_benchmark.py`` via importlib so this module works when
    executed as a bare file (e.g. ``earnings_comparison.py``) and so ``@dataclass``
    sees a real ``sys.modules`` entry during load.
    """
    name = "lifecycle_model.gkos_benchmark"
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gkos_benchmark.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load gkos_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_gb = _load_gkos_benchmark()

_earnings_log_count = 0


def params_from_dict(params: dict | None) -> _gb.GKOSBenchmarkParams:
    return _gb.params_from_dict(params)


def compute_age_profile(ages, params):
    """
    Additive log component g(t) + intercept (Table D.2), for debug overlays.

    ``earnings_iqr_debug`` uses baseline * exp(profile) as a deterministic track.
    """
    p = _gb.params_from_dict(params if params is not None else {})
    ages = np.asarray(ages, dtype=float)
    t_norm = (ages - 24.0) / 10.0
    return p.g_a0 + p.g_a1 * t_norm + p.g_a2 * (t_norm**2) + p.additional_log_intercept


def baseline_for_median_at_start(annual_income_real, age_start, age_end, params):
    """
    Multiplicative baseline so median first-year positive earnings match
    ``annual_income_real`` under the benchmark process.
    """
    p = _gb.params_from_dict(params)
    med = _gb.median_first_year_positive(p, int(age_start), t_years=1)
    return float(annual_income_real) / med


def arc_percent_changes(earnings):
    """
    Arc-percent changes (GKOS): Δ_arc Y = (Y_{t+1} - Y_t) / ((Y_{t+1} + Y_t)/2).
    Returns array of length len(earnings)-1. Handles zeros by returning NaN where sum is 0.
    """
    y = np.asarray(earnings, dtype=float)
    y_next = y[1:]
    y_curr = y[:-1]
    mid = (y_next + y_curr) / 2.0
    out = np.full_like(mid, np.nan)
    np.divide(y_next - y_curr, mid, out=out, where=mid > 0)
    return out


def simulate_single_earnings_path(age_start, age_end, baseline_earnings, params, rng):
    """
    One earnings path: Y_t = baseline * (1-ν) exp(g(t)+α+βt+z+ε), benchmark Table IV.
    """
    age_start = int(age_start)
    age_end = int(age_end)
    t_years = age_end - age_start + 1
    p = _gb.params_from_dict(params)
    shocks = _gb.make_shocks(1, t_years, rng)
    _, y, ages = _gb.simulate_gkos(
        params=p,
        n_workers=1,
        t_years=t_years,
        age_start=age_start,
        shocks=shocks,
    )
    earnings = y[0].astype(float) * float(baseline_earnings)

    global _earnings_log_count
    if _earnings_log_count < 1:
        _earnings_log_count += 1
        try:
            _lp = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                ".debug",
                "debug.log",
            )
            _d = {
                "years": t_years,
                "typical_worker": p.typical_worker,
                "additional_log_intercept": p.additional_log_intercept,
                "earnings_min": float(np.min(earnings)),
                "earnings_max": float(np.max(earnings)),
                "earnings_mean": float(np.mean(earnings)),
            }
            with open(_lp, "a", encoding="utf-8") as _f:
                _f.write(
                    json.dumps(
                        {
                            "id": "earnings_path_one",
                            "timestamp": 0,
                            "location": "earnings.py:simulate_single_earnings_path",
                            "message": "first path stats",
                            "data": _d,
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass

    return earnings, ages


def simulate_earnings_path_with_inflation(
    age_start, age_end, baseline_earnings, inflation_rates, params, rng
):
    """
    Simulate earnings path then convert to nominal using inflation_rates.
    """
    real_earnings, ages = simulate_single_earnings_path(
        age_start, age_end, baseline_earnings=baseline_earnings, params=params, rng=rng
    )
    nominal_earnings = np.zeros(len(ages))
    nominal_earnings[0] = real_earnings[0]
    cumulative_inflation = 1.0
    n_years = len(ages)
    n_inflation_rates = len(inflation_rates)
    for i in range(1, n_years):
        if i - 1 < n_inflation_rates:
            cumulative_inflation *= 1 + inflation_rates[i - 1]
        nominal_earnings[i] = real_earnings[i] * cumulative_inflation
    return nominal_earnings, real_earnings, ages
