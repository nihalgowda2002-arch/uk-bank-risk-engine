"""Asset and liability management: deposit behaviour and the structural hedge.

Models the runoff of core deposits as market rates move, then builds a rolling
ladder of SONIA receiver swaps and measures how much of the net interest income
volatility across macro scenarios that ladder removes.

Two modelling choices are worth stating, because both materially change the
answer and neither is specified in the original plan:

1. The hedge notional is a share of the *rate-insensitive* portion of the
   deposit base, ``deposits * (1 - deposit_beta)``, not of the whole book. It
   is the balances that do not reprice which create the structural exposure,
   so hedging the full deposit base would over-hedge.
2. The ladder is assumed mature at the start of the simulation, with tranches
   struck over the preceding ``tenor`` years at ``historical_rate``. A ladder
   built from nothing has almost no effect in its first years, which flatters
   the unhedged case and understates what a real treasury book does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAMBDA_BASE = 0.05   # annual runoff of core deposits when rates are unchanged
GAMMA = 0.60         # sensitivity of runoff to the market minus deposit rate gap
DEPOSIT_BETA = 0.35  # share of a market rate move passed on to depositors


def deposit_balance(d0: float, years: np.ndarray, market_rate: np.ndarray,
                    deposit_rate: np.ndarray, lambda_base: float = LAMBDA_BASE,
                    gamma: float = GAMMA) -> np.ndarray:
    """Behavioural deposit balance under exponential decay.

    ``D(t) = D0 * exp(-lambda * t)`` with
    ``lambda = lambda_base + gamma * max(market_rate - deposit_rate, 0)``.
    The gap is floored at zero so that a deposit book never grows through this
    channel: depositors leave when they can do better elsewhere, but they do
    not arrive merely because they cannot.
    """
    gap = np.maximum(np.asarray(market_rate) - np.asarray(deposit_rate), 0.0)
    lam = lambda_base + gamma * gap
    return float(d0) * np.exp(-lam * np.asarray(years, dtype=float))


def deposit_rate_from_beta(market_rate: np.ndarray,
                           beta: float = DEPOSIT_BETA) -> np.ndarray:
    """Rate paid to depositors, passing through ``beta`` of the market rate."""
    return np.maximum(beta * np.asarray(market_rate, dtype=float), 0.0)


def swap_ladder(notional: float, tenor: int, market_rate: np.ndarray,
                historical_rate: float | None = None) -> pd.DataFrame:
    """A mature rolling ladder of receiver swaps.

    One tranche of ``notional / tenor`` matures and is replaced each year. At
    the start of the simulation the ladder already holds ``tenor`` tranches
    struck over the preceding years at ``historical_rate``. Each tranche
    receives the fixed rate at which it was struck and pays the prevailing
    floating rate, so the ladder receives a moving average of past rates and
    pays today's rate. That averaging is the entire mechanism.
    """
    market = np.asarray(market_rate, dtype=float)
    n = len(market)
    if historical_rate is None:
        historical_rate = float(market[0])
    tranche = notional / tenor

    struck = [float(historical_rate)] * tenor  # oldest first
    receive, pay, avg_fixed = np.zeros(n), np.zeros(n), np.zeros(n)

    for t in range(n):
        receive[t] = tranche * float(np.sum(struck))
        pay[t] = notional * market[t]
        avg_fixed[t] = float(np.mean(struck))
        struck.pop(0)                    # the oldest tranche matures
        struck.append(float(market[t]))  # and is replaced at today's rate

    return pd.DataFrame({
        "swap_receive": receive,
        "swap_pay": pay,
        "swap_net": receive - pay,
        "average_fixed_rate": avg_fixed,
    })


def simulate_nii(macro_annual: pd.DataFrame, scenario: str,
                 loans_0: float = 60e9, deposits_0: float = 55e9,
                 loan_spread: float = 0.0150, hedge_ratio: float = 0.80,
                 tenor: int = 5, deposit_beta: float = DEPOSIT_BETA,
                 historical_rate: float = 0.0300) -> pd.DataFrame:
    """Net interest income with and without the structural hedge.

    Assets are floating and earn the bank rate plus a fixed spread. Deposits
    pay ``deposit_beta`` of the bank rate and run off behaviourally. The hedge
    receives fixed on ``hedge_ratio`` of the rate-insensitive deposit base.
    """
    macro = (macro_annual[macro_annual["scenario"] == scenario]
             .sort_values("year").reset_index(drop=True))
    if macro.empty:
        raise ValueError(f"no macro path for scenario {scenario!r}")

    years = macro["year"].to_numpy(dtype=float)
    market = macro["bank_rate"].to_numpy()
    dep_rate = deposit_rate_from_beta(market, deposit_beta)
    balances = deposit_balance(deposits_0, years - years[0], market, dep_rate)

    asset_income = loans_0 * (market + loan_spread)
    deposit_cost = balances * dep_rate
    nii_unhedged = asset_income - deposit_cost

    hedge_notional = deposits_0 * (1.0 - deposit_beta) * hedge_ratio
    ladder = swap_ladder(hedge_notional, tenor, market, historical_rate)
    nii_hedged = nii_unhedged + ladder["swap_net"].to_numpy()

    return pd.DataFrame({
        "year": years.astype(int),
        "scenario": scenario,
        "bank_rate": market,
        "deposit_rate": dep_rate,
        "deposit_balance": balances,
        "hedge_notional": hedge_notional,
        "average_fixed_rate": ladder["average_fixed_rate"].to_numpy(),
        "nii_unhedged": nii_unhedged,
        "swap_net": ladder["swap_net"].to_numpy(),
        "nii_hedged": nii_hedged,
    })


def nii_volatility(macro_annual: pd.DataFrame,
                   scenarios=("base", "upside", "downside"), **kwargs) -> pd.DataFrame:
    """Spread of net interest income across scenarios, hedged and unhedged.

    A structural hedge is judged by how much it narrows the range of income
    across scenarios, not by whether it raises income in any single one. In a
    rising rate path the hedge loses money, and that is intended behaviour.
    """
    frames = [simulate_nii(macro_annual, s, **kwargs) for s in scenarios]
    allof = pd.concat(frames, ignore_index=True)
    rows = []
    for year, grp in allof.groupby("year"):
        un = grp["nii_unhedged"].max() - grp["nii_unhedged"].min()
        he = grp["nii_hedged"].max() - grp["nii_hedged"].min()
        rows.append({"year": int(year), "unhedged_range": un, "hedged_range": he,
                     "reduction_pct": 100.0 * (1.0 - he / un) if un > 0 else np.nan})
    return pd.DataFrame(rows)
