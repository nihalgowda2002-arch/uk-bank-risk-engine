"""Synthetic UK macroeconomic scenarios and retail mortgage portfolio.

Nothing in this module is real data. The distributions are chosen to be of a
plausible order of magnitude for a UK retail mortgage book, so that the
downstream engines can be exercised end to end and their behaviour inspected.
They are not calibrated to any bank's book and should not be read as estimates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SCENARIOS = ("base", "upside", "downside")

# Scenario weights used for the probability-weighted ECL required by IFRS 9.
SCENARIO_WEIGHTS: dict[str, float] = {"base": 0.50, "upside": 0.20, "downside": 0.30}

# Terminal levels each scenario moves towards, and the starting point.
_SCENARIO_SPEC = {
    #            bank rate, unemployment, annual HPI growth
    "base":     {"rate": 0.0375, "unemp": 0.040, "hpi": 0.020},
    "upside":   {"rate": 0.0300, "unemp": 0.035, "hpi": 0.050},
    "downside": {"rate": 0.0600, "unemp": 0.075, "hpi": -0.120},
}
_START = {"rate": 0.0475, "unemp": 0.043, "hpi": 0.010}


def generate_macro_scenarios(n_years: int = 5, seed: int = 20260818) -> pd.DataFrame:
    """Quarterly macroeconomic paths for each scenario.

    Paths move from the current level to the scenario's terminal level over the
    first two years and then hold, with a small amount of noise so that the
    series are not perfectly smooth.

    Returns a long DataFrame with columns
    ``scenario, quarter, year, bank_rate, unemployment, hpi_growth``.
    """
    rng = np.random.default_rng(seed)
    n_q = n_years * 4
    ramp = np.clip(np.arange(1, n_q + 1) / 8.0, 0.0, 1.0)  # two-year convergence

    frames = []
    for name in SCENARIOS:
        spec = _SCENARIO_SPEC[name]
        rows = {"scenario": name, "quarter": np.arange(1, n_q + 1)}
        for key, col in (("rate", "bank_rate"), ("unemp", "unemployment"),
                         ("hpi", "hpi_growth")):
            path = _START[key] + ramp * (spec[key] - _START[key])
            path = path + rng.normal(0.0, 0.0015, n_q)
            if key in ("rate", "unemp"):
                path = np.maximum(path, 0.0)
            rows[col] = path
        frame = pd.DataFrame(rows)
        frame["year"] = np.ceil(frame["quarter"] / 4).astype(int)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def annual_macro(macro: pd.DataFrame) -> pd.DataFrame:
    """Collapse the quarterly paths to annual averages, with changes from t=0.

    ``d_unemployment`` is the change in the unemployment rate relative to the
    starting level, in percentage points expressed as a decimal.
    ``hpi_index`` is the cumulative house price index, starting at 1.0.
    """
    out = (macro.groupby(["scenario", "year"], as_index=False)
                .agg(bank_rate=("bank_rate", "mean"),
                     unemployment=("unemployment", "mean"),
                     hpi_growth=("hpi_growth", "mean"))
                .sort_values(["scenario", "year"], ignore_index=True))
    out["d_unemployment"] = out["unemployment"] - _START["unemp"]
    out["hpi_index"] = (out.groupby("scenario")["hpi_growth"]
                           .transform(lambda g: (1.0 + g).cumprod()))
    return out


def generate_loan_portfolio(n: int = 10_000, seed: int = 20260818) -> pd.DataFrame:
    """A synthetic UK residential mortgage portfolio.

    Fields follow the plan: balance, current LTV, borrower income, credit score
    and interest type. Origination PD is derived from the credit score by a
    logistic map, so that score and default risk are consistently related.
    """
    rng = np.random.default_rng(seed)

    balance = np.clip(rng.lognormal(mean=np.log(250_000), sigma=0.45, size=n),
                      150_000, 750_000)
    ltv = np.clip(rng.normal(0.72, 0.12, n), 0.50, 0.95)
    income = np.clip(rng.lognormal(mean=np.log(48_000), sigma=0.45, size=n),
                     25_000, 150_000)
    score = np.clip(rng.normal(700, 70, n), 300, 850).round().astype(int)
    interest_type = rng.choice(["fixed_2y", "fixed_5y", "svr"], size=n,
                               p=[0.45, 0.40, 0.15])
    term_remaining = rng.integers(5, 26, size=n)

    # Origination twelve-month PD, decreasing in credit score.
    # A score of 850 maps to roughly 0.15%, a score of 300 to roughly 12%.
    z = (score - 300.0) / 550.0
    pd_12m = 0.0015 + (0.12 - 0.0015) * np.exp(-3.2 * z)

    # Debt service burden.
    dsr = np.clip((balance * 0.05) / income, 0.05, 1.50)

    # Loan-level sensitivity to the macroeconomy. Highly geared borrowers with
    # a heavy debt service burden respond far more to an unemployment shock or
    # a fall in house prices than a low-LTV borrower with slack income, so a
    # single economy-wide shift would understate the dispersion of outcomes.
    sensitivity = np.clip(
        1.0 + 3.0 * (ltv - 0.72) + 1.2 * (dsr - 0.30), 0.25, 3.00)

    return pd.DataFrame({
        "loan_id": [f"L{i:06d}" for i in range(n)],
        "balance": balance.round(2),
        "current_ltv": ltv.round(4),
        "borrower_income": income.round(2),
        "credit_score": score,
        "interest_type": interest_type,
        "term_remaining": term_remaining,
        "pd_12m_origination": pd_12m.round(6),
        "debt_service_ratio": dsr.round(4),
        "macro_sensitivity": sensitivity.round(4),
        "days_past_due": np.where(rng.random(n) < 0.015,
                                  rng.integers(1, 180, n), 0),
    })


if __name__ == "__main__":  # pragma: no cover
    macro = generate_macro_scenarios()
    loans = generate_loan_portfolio()
    macro.to_csv("data/macro_scenarios.csv", index=False)
    loans.to_csv("data/loan_portfolio.csv", index=False)
    print(f"macro scenarios: {macro.shape}, loans: {loans.shape}")
    print(annual_macro(macro).head(10).to_string(index=False))
