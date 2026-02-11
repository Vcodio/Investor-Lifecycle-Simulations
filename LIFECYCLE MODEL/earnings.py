"""
GKOS Earnings Model — Guvenen, Karahan, Ozkan, Song (2015) Benchmark.

Implements the Benchmark stochastic earnings process (Table IV):
- Y_tilde = (1-ν)*exp(g(t) + α_i + β_i*t + z_it + ε_it)
- g(t): quadratic age profile peaking at AGE_PEAK
- z_it: AR(1) with mixture-of-normals η; initial z_i0 ~ N(0, σ_z0²)
- ε_it: mixture of two normals (high kurtosis, negative skewness)
- ν_it: nonemployment with p_ν(t,z) = exp(ξ)/(1+exp(ξ)), ξ = a + b*t + c*z + d*z*t (scarring)
- HIP: (α_i, β_i) ~ bivariate normal
"""

import numpy as np
import os

_earnings_log_count = 0


def _get(params, key, default=None):
    """Get param with backward compatibility for old param names."""
    if key in params:
        return params[key]
    return default


def generate_persistent_shock_mixture(n_samples, params, rng):
    """
    Persistent shocks η: mixture of two normals (Model 8).
    Unfavorable (p1≈0.219): μ1=-0.15, σ1=0.46; Favorable: μ2=0.041, σ2=0.14.
    Targets skewness ≈ -1.10, kurtosis ≈ 8.
    """
    p1 = _get(params, 'ETA_P1', params.get('MIXTURE_PROB_TAIL', 0.219))
    if 'ETA_MU1' in params:
        mu1 = params['ETA_MU1']
        sigma1 = params['ETA_SIGMA1']
        mu2 = params['ETA_MU2']
        sigma2 = params['ETA_SIGMA2']
        ind = rng.random(size=n_samples) < p1
        eta1 = rng.normal(mu1, sigma1, size=n_samples)
        eta2 = rng.normal(mu2, sigma2, size=n_samples)
        return np.where(ind, eta1, eta2)

    component_indicator = rng.binomial(1, p1, size=n_samples)
    normal_shocks = rng.normal(0, params.get('SIGMA_ETA', 0.15), size=n_samples)
    tail_std = params.get('SIGMA_ETA', 0.15) * np.sqrt(params.get('TAIL_VARIANCE_MULTIPLIER', 25.0))
    tail_mean = params.get('SKEWNESS_PARAM', -2.5) * params.get('SIGMA_ETA', 0.15)
    tail_shocks = rng.normal(tail_mean, tail_std, size=n_samples)
    return np.where(component_indicator == 0, normal_shocks, tail_shocks)


def generate_transitory_shock(n_samples, params, rng):
    """
    Transitory shocks ε: mixture of two normals (GKOS 2015 Table IV).
    Unfavorable (prob p_ε=0.118): N(μ_ε,1, σ_ε,1); Favorable: N(μ_ε,2, σ_ε,2).
    """
    if 'EPS_P1' in params:
        p1 = params['EPS_P1']
        mu1 = params.get('EPS_MU1', -0.554)
        s1 = params['EPS_SIGMA1']
        mu2 = params.get('EPS_MU2', 0.074)
        s2 = params['EPS_SIGMA2']
        ind = rng.random(size=n_samples) < p1
        e1 = rng.normal(mu1, s1, size=n_samples)
        e2 = rng.normal(mu2, s2, size=n_samples)
        return np.where(ind, e1, e2)
    scale = params.get('SIGMA_EPSILON', 0.05) / np.sqrt(2)
    return rng.laplace(0, scale, size=n_samples)


def compute_age_profile(ages, params):
    """
    Deterministic age-earnings profile g(t) from GKOS (2015).
    g(t) = a * (t - AGE_PEAK)^2 with a < 0 so that earnings peak exactly at AGE_PEAK.
    No linear term and no demeaning — peak is at AGE_PEAK by construction.
    """
    age_peak = params['AGE_PEAK']
    curvature = params.get('AGE_PROFILE_A', -0.0005)
    age_centered = ages - age_peak
    profile = curvature * (age_centered ** 2)
    return profile


def baseline_for_median_at_start(annual_income_real, age_start, age_end, params):
    """
    Return baseline_earnings so that median earnings at age_start equals annual_income_real.
    Median individual has α=0, β=0, z=0, ε=0: Y = baseline * exp(g(age_start)).
    """
    ages = np.arange(age_start, age_end + 1)
    profile = compute_age_profile(ages, params)
    return float(annual_income_real) / np.exp(profile[0])


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


def _p_nu_nonemployment(age, z_t, params):
    """
    Probability of nonemployment shock from GKOS (2015): p_ν(t,z) = exp(ξ)/(1+exp(ξ)),
    ξ = a + b*t + c*z + d*z*t (scarring: low z and interaction increase risk).
    """
    a = _get(params, 'NU_A', -3.12)
    b = _get(params, 'NU_B', 0.005)
    c = _get(params, 'NU_C', -1.15)
    d = _get(params, 'NU_D', -0.015)
    xi = a + b * age + c * z_t + d * z_t * age

    p = 1.0 / (1.0 + np.exp(-xi))
    return float(np.clip(p, 0.0, 1.0))


def _draw_hip(params, rng):
    """
    Draw (α_i, β_i) from bivariate normal per GKOS (2015): σ_α, σ_β, corr(α,β).
    Returns (alpha_i, beta_i) for use as α_i + β_i * t.
    """
    sigma_alpha = _get(params, 'SIGMA_ALPHA', 0.0)
    sigma_beta = _get(params, 'SIGMA_BETA', 0.0)
    corr_ab = _get(params, 'HIP_CORR_AB', -0.01)
    if sigma_alpha <= 0 and sigma_beta <= 0:
        return 0.0, 0.0
    if sigma_beta <= 0:
        return (rng.normal(0, sigma_alpha), 0.0)
    if sigma_alpha <= 0:
        return (0.0, rng.normal(0, sigma_beta))
    cov_ab = corr_ab * sigma_alpha * sigma_beta
    cov = np.array([[sigma_alpha**2, cov_ab], [cov_ab, sigma_beta**2]])
    alpha_i, beta_i = rng.multivariate_normal([0.0, 0.0], cov)
    return float(alpha_i), float(beta_i)


def simulate_single_earnings_path(age_start, age_end, baseline_earnings, params, rng):
    """
    Simulate a single individual's earnings path (GKOS 2015 Benchmark).
    Y_tilde = (1-ν)*exp(g(t)+α_i+β_i*t+z_it+ε_it); g(t) peaks at AGE_PEAK.
    - Persistent η: mixture of two normals (Table IV)
    - Transitory ε: mixture of two normals
    - HIP: (α_i, β_i) ~ bivariate normal
    - Nonemployment: p_ν(t,z) = exp(ξ)/(1+exp(ξ)), ξ = a+b*t+c*z+d*z*t
    """
    ages = np.arange(age_start, age_end + 1)
    years = len(ages)
    common_profile = compute_age_profile(ages, params)


    alpha_i, beta_i = _draw_hip(params, rng)
    years_from_start = np.arange(years, dtype=float)
    hip_component = alpha_i + beta_i * years_from_start
    age_profile = common_profile + hip_component


    sigma_z0 = _get(params, 'SIGMA_Z0', 0.272)
    z = np.zeros(years)
    z[0] = rng.normal(0, sigma_z0)


    persistent_shocks = generate_persistent_shock_mixture(years - 1, params, rng)
    transitory_shocks = generate_transitory_shock(years, params, rng)


    rho = params.get('RHO', 0.97)
    for t in range(1, years):
        z[t] = rho * z[t - 1] + persistent_shocks[t - 1]


    nonemployment = np.zeros(years)
    for t in range(years):
        age_t = ages[t]
        z_t = z[t]
        p_nu = _p_nu_nonemployment(age_t, z_t, params)
        if rng.random() < p_nu:
            nonemployment[t] = 1.0


    log_earnings = z + age_profile + transitory_shocks

    log_earnings[nonemployment > 0.5] = -20.0

    earnings = np.exp(log_earnings) * baseline_earnings

    global _earnings_log_count
    if _earnings_log_count < 1:
        _earnings_log_count += 1
        try:
            _nz = int(np.sum(nonemployment > 0.5))
            _lp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.cursor', 'debug.log')
            import json
            _d = {'years': years, 'alpha_i': float(alpha_i), 'g_i': float(g_i), 'n_nonemployment': _nz, 'log_earnings_min': float(np.min(log_earnings)), 'log_earnings_max': float(np.max(log_earnings)), 'earnings_min': float(np.min(earnings)), 'earnings_max': float(np.max(earnings)), 'earnings_mean': float(np.mean(earnings)), 'hypothesisId': 'H1_H4'}
            with open(_lp, 'a', encoding='utf-8') as _f:
                _f.write(json.dumps({'id': 'earnings_path_one', 'timestamp': 0, 'location': 'earnings.py:simulate_single_earnings_path', 'message': 'first path stats', 'data': _d, 'runId': 'debug'}) + '\n')
        except Exception:
            pass

    return earnings, ages


def simulate_earnings_path_with_inflation(age_start, age_end, baseline_earnings, inflation_rates, params, rng):
    """
    Simulate earnings path (Model 8) then convert to nominal using inflation_rates.
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
            cumulative_inflation *= (1 + inflation_rates[i - 1])
        nominal_earnings[i] = real_earnings[i] * cumulative_inflation
    return nominal_earnings, real_earnings, ages
