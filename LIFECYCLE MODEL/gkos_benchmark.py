"""
GKOS benchmark earnings process (SR 710 / 2019; Table IV model 8; Table D.2).

Shared by ``earnings.py`` (lifecycle simulations) and the standalone
``labor_income_tests/GKOS_benchmark_sim.py`` plot driver.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GKOSBenchmarkParams:
    """Table IV, Model (8), plus Table D.2 column (8) for g(t) and p_nu."""

    rho: float = 0.958
    p_z: float = 0.219
    mu_eta1: float = -0.147
    sigma_eta1: float = 0.457
    sigma_eta2: float = 0.139
    sigma_z0: float = 0.667
    lambda_nu: float = 0.001
    p_eps: float = 0.126
    mu_eps1: float = 0.236
    sigma_eps1: float = 0.343
    sigma_eps2: float = 0.063
    sigma_alpha: float = 0.298
    sigma_beta: float = 0.185
    corr_alpha_beta: float = 0.976
    g_a0: float = 2.6291
    g_a1: float = 0.7300
    g_a2: float = -0.1692
    a_nu: float = -3.2131
    b_nu: float = -1.0235
    c_nu: float = -3.2602
    d_nu: float = -2.1656
    additional_log_intercept: float = 0.0
    typical_worker: bool = False

    @property
    def mu_eta2(self) -> float:
        p = self.p_z
        return -(p * self.mu_eta1) / (1.0 - p)

    @property
    def mu_eps2(self) -> float:
        p = self.p_eps
        return -(p * self.mu_eps1) / (1.0 - p)


def params_from_dict(params: dict | None) -> GKOSBenchmarkParams:
    """Build GKOSBenchmarkParams from a config dict, falling back to dataclass defaults."""
    if not params:
        return GKOSBenchmarkParams()
    d = GKOSBenchmarkParams()  # start from canonical defaults
    def _f(key, default): return float(params.get(key, default))
    def _b(key, default): return bool(params.get(key, default))
    return GKOSBenchmarkParams(
        typical_worker=_b("typical_worker", d.typical_worker),
        additional_log_intercept=_f("additional_log_intercept", d.additional_log_intercept),
        rho=_f("rho", d.rho),
        p_z=_f("p_z", d.p_z),
        mu_eta1=_f("mu_eta1", d.mu_eta1),
        sigma_eta1=_f("sigma_eta1", d.sigma_eta1),
        sigma_eta2=_f("sigma_eta2", d.sigma_eta2),
        sigma_z0=_f("sigma_z0", d.sigma_z0),
        lambda_nu=_f("lambda_nu", d.lambda_nu),
        p_eps=_f("p_eps", d.p_eps),
        mu_eps1=_f("mu_eps1", d.mu_eps1),
        sigma_eps1=_f("sigma_eps1", d.sigma_eps1),
        sigma_eps2=_f("sigma_eps2", d.sigma_eps2),
        sigma_alpha=_f("sigma_alpha", d.sigma_alpha),
        sigma_beta=_f("sigma_beta", d.sigma_beta),
        corr_alpha_beta=_f("corr_alpha_beta", d.corr_alpha_beta),
        g_a0=_f("g_a0", d.g_a0),
        g_a1=_f("g_a1", d.g_a1),
        g_a2=_f("g_a2", d.g_a2),
        a_nu=_f("a_nu", d.a_nu),
        b_nu=_f("b_nu", d.b_nu),
        c_nu=_f("c_nu", d.c_nu),
        d_nu=_f("d_nu", d.d_nu),
    )


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def make_shocks(n_workers: int, t_years: int, rng: np.random.Generator) -> dict:
    return {
        "hip": rng.standard_normal((n_workers, 2)),
        "z0": rng.standard_normal(n_workers),
        "eta_mix": rng.random((n_workers, t_years)),
        "eta_1": rng.standard_normal((n_workers, t_years)),
        "eta_2": rng.standard_normal((n_workers, t_years)),
        "eps_mix": rng.random((n_workers, t_years)),
        "eps_1": rng.standard_normal((n_workers, t_years)),
        "eps_2": rng.standard_normal((n_workers, t_years)),
        "nu_mix": rng.random((n_workers, t_years)),
        "nu_exp": rng.exponential(1.0, (n_workers, t_years)),
    }


def simulate_gkos(
    params: GKOSBenchmarkParams | None = None,
    n_workers: int = 50_000,
    t_years: int = 36,
    age_start: int = 25,
    seed: int = 42,
    rng: np.random.Generator | None = None,
    shocks: dict | None = None,
    y_floor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate eqs. (2)-(8). Returns (log_y, y_levels, ages).

    If ``shocks`` is None, draws using ``rng`` or ``np.random.default_rng(seed)``.
    """
    if params is None:
        params = GKOSBenchmarkParams()
    if shocks is None:
        rng = rng if rng is not None else np.random.default_rng(seed)
        shocks = make_shocks(n_workers, t_years, rng)
    else:
        n_workers = int(shocks["z0"].shape[0])
        t_years = int(shocks["eta_mix"].shape[1])

    ages = np.arange(age_start, age_start + t_years, dtype=float)
    t_norm = (ages - 24.0) / 10.0
    g_t = params.g_a0 + params.g_a1 * t_norm + params.g_a2 * (t_norm**2)

    sig_b = float(params.sigma_beta)
    cov_ab = params.corr_alpha_beta * params.sigma_alpha * sig_b
    cov = np.array(
        [[params.sigma_alpha**2, cov_ab], [cov_ab, sig_b**2]], dtype=float
    )
    chol = np.linalg.cholesky(cov)
    hip = shocks["hip"] @ chol.T
    alpha = hip[:, 0]
    beta = hip[:, 1]

    z = np.zeros((n_workers, t_years))
    if params.typical_worker:
        alpha = np.zeros(n_workers)
        beta = np.zeros(n_workers)
        z[:, 0] = 0.0
    else:
        z[:, 0] = shocks["z0"] * params.sigma_z0
    y = np.zeros((n_workers, t_years))

    p_z = params.p_z
    p_eps = params.p_eps
    mu_eta2 = params.mu_eta2
    mu_eps2 = params.mu_eps2
    lam = params.lambda_nu
    scale_exp = 1.0 / lam if lam > 0 else np.inf

    for t in range(t_years):
        if t > 0:
            draw = shocks["eta_mix"][:, t] < p_z
            eta = np.where(
                draw,
                params.mu_eta1 + shocks["eta_1"][:, t] * params.sigma_eta1,
                mu_eta2 + shocks["eta_2"][:, t] * params.sigma_eta2,
            )
            z[:, t] = params.rho * z[:, t - 1] + eta
        draw_eps = shocks["eps_mix"][:, t] < p_eps
        eps = np.where(
            draw_eps,
            params.mu_eps1 + shocks["eps_1"][:, t] * params.sigma_eps1,
            mu_eps2 + shocks["eps_2"][:, t] * params.sigma_eps2,
        )
        xi = (
            params.a_nu
            + params.b_nu * t_norm[t]
            + params.c_nu * z[:, t]
            + params.d_nu * z[:, t] * t_norm[t]
        )
        p_nu = logistic(xi)
        hit = shocks["nu_mix"][:, t] < p_nu
        raw_exp = shocks["nu_exp"][:, t] * scale_exp
        nu = np.where(hit, np.minimum(1.0, raw_exp), 0.0)
        x = (
            g_t[t]
            + alpha
            + beta * t_norm[t]
            + z[:, t]
            + eps
            + params.additional_log_intercept
        )
        x = np.clip(x, -80.0, 80.0)
        y[:, t] = (1.0 - nu) * np.exp(x)
        y[:, t] = np.where(np.isfinite(y[:, t]), y[:, t], 0.0)

    log_y = np.log(np.maximum(y, y_floor))
    return log_y, y, ages


_median_y0_cache: dict[tuple[bool, float, int], float] = {}


def median_first_year_positive(
    p: GKOSBenchmarkParams,
    age_start: int,
    t_years: int = 1,
    n_draw: int = 8000,
    cache_seed: int = 12345,
) -> float:
    """Median year-0 earnings among positive values; used to scale baseline."""
    key = (p.typical_worker, round(p.additional_log_intercept, 6), age_start)
    if key in _median_y0_cache:
        return _median_y0_cache[key]
    rng = np.random.default_rng(cache_seed)
    shocks = make_shocks(n_draw, t_years, rng)
    _, y, _ = simulate_gkos(
        params=p,
        n_workers=n_draw,
        t_years=t_years,
        age_start=age_start,
        shocks=shocks,
    )
    y0 = y[:, 0]
    pos = y0[y0 > 0.0]
    med = float(np.median(pos)) if pos.size > n_draw // 4 else float(np.median(y0))
    med = max(med, 1.0)
    _median_y0_cache[key] = med
    return med
