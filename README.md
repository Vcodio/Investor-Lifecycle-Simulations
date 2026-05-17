# Investor Lifecycle Simulations

**Version 10.0.7** — Streamlit app, GKOS labor income, block bootstrap, and optional amortization-based withdrawals.

Simulate how an investor might fare over a lifetime: required wealth to retire at a given success rate, distribution of feasible retirement ages, and withdrawal sustainability under uncertain returns and earnings. Amounts in the app are in **real** (inflation-adjusted) terms unless noted otherwise.

## How it works

**Stage 1 — Required principal**  
For each candidate retirement age, nested Monte Carlo runs estimate the portfolio size needed to hit your target success rate (e.g. 95%) in the withdrawal phase.

**Stage 2 — Accumulation**  
Thousands of outer paths simulate working years: savings, GKOS stochastic earnings, portfolio returns (parametric Bates or historical block bootstrap), and optional Social Security. Each path stops when wealth reaches the Stage 1 target, producing a distribution of retirement ages.

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Optional but recommended for speed:

```bash
./build_cython.sh    # Linux/macOS
# or build_cython.bat on Windows
```

### 2. Run the Streamlit app

```bash
python launch_app.py
```

Open [http://localhost:8501](http://localhost:8501). The launcher finds `streamlit` on your PATH (including pipx installs).

### 3. Run from the command line (no UI)

```bash
python "LIFECYCLE MODEL/run_simulation.py"
```

## Project layout

| Path | Purpose |
|------|---------|
| `app.py` | Streamlit UI: lifecycle simulation + parametric model estimation |
| `launch_app.py` | Starts Streamlit with dependency checks |
| `LIFECYCLE MODEL/` | Core engine: simulation, bootstrap, GKOS earnings, utility, plots |
| `LIFECYCLE MODEL/gkos_benchmark.py` | Table IV GKOS process (shared with benchmark script) |
| `labor_income_tests/GKOS_benchmark_sim.py` | Standalone GKOS earnings fan-chart benchmark |
| `Parametric Model (Unfinished)/` | HMM regime detection and Bates/MJD moment matching |
| `data/` | Bootstrap CSVs, regime labels, mortality tables |

## Key features

- **GKOS earnings** — Guvenen, Karahan, Ozkan & Song (2019) Table IV model; median starting earnings scaled to your `annual_income_real`
- **Returns** — Bates stochastic vol + jumps (parametric) or overlapping block bootstrap from historical portfolio + inflation series
- **Withdrawals** — Fixed real spending or amortization-based withdrawals with optional bequest
- **Cython** — Accelerated return and MLE paths; falls back to pure Python if extensions are not built
- **Plots** — Plotly charts in the app (install `plotly`); matplotlib fallback
- **Utility** — Optional CRRA utility and certainty-equivalent consumption (sidebar toggle)

## Main parameters (sidebar)

| Parameter | Meaning |
|-----------|---------|
| `initial_age` | Start of working life |
| `death_age` | End of simulation (or stochastic mortality table) |
| `initial_portfolio` | Starting wealth |
| `annual_income_real` | Current annual labor income (real $) |
| `spending_real` | Target annual retirement spending (real $) |
| `savings_rate` | Fraction of income saved while working |
| `num_outer` / `num_nested` | Outer lifecycle paths / nested runs per retirement age |
| `success_target` | Required success rate for principal lookup (e.g. 0.95) |
| `use_block_bootstrap` | Use historical blocks instead of Bates draws |
| `use_amortization` | Amortization-based withdrawal path in retirement |

GKOS structural parameters live under **Advanced → Earnings Dynamics (GKOS)** and match `GKOSBenchmarkParams` in `gkos_benchmark.py` (`typical_worker=True` is the Cederburg et al. base case).

## GKOS benchmark

To reproduce the standalone earnings fan chart (ages 25–60, 200k workers):

```bash
python labor_income_tests/GKOS_benchmark_sim.py
```

Output: `labor_income_tests/gkos_levels.png` and console diagnostics (non-employment rate, percentiles at peak age).

The app’s **Labor Income Dynamics** tab runs the same engine via `simulate_gkos`, scaled to your income and starting age.

## Parametric model tab

The second tab supports HMM regime detection on return series and regime-conditional Bates (or MJD) parameter estimation. Results can inform `config.params` or bootstrap inputs when using the parametric return model.

## Roadmap (selected)

1. ~~Regime-switching return model~~ — partial (estimation modules; full integration in progress)
2. ~~Amortization-based withdrawals~~ — available; success metrics still evolving
3. Forecast error on expected returns for amortization
4. Utility of consumption and bequest in accumulation and retirement
5. Couple mortality / longevity risk
6. Taxes, fees, glide paths, annuities

## License

See [LICENSE](LICENSE).
