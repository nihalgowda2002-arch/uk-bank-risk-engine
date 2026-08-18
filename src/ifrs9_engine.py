"""Unit tests for the IFRS 9 expected credit loss engine."""
import numpy as np
import pytest

from src.data_generator import (SCENARIO_WEIGHTS, annual_macro,
                                generate_loan_portfolio, generate_macro_scenarios)
from src.ifrs9_engine import (DEFAULT_DPD, SICR_PD_RATIO, LGD_CAP, LGD_FLOOR,
                              assign_stage, amortising_ead, haircut_lgd,
                              macro_adjusted_pd, scenario_ecl, weighted_ecl)


@pytest.fixture(scope="module")
def portfolio():
    return generate_loan_portfolio(n=500, seed=1)


@pytest.fixture(scope="module")
def macro():
    return annual_macro(generate_macro_scenarios(seed=1))


# --------------------------------------------------------------- PD behaviour
def test_pd_stays_a_probability():
    pd0 = np.array([1e-6, 0.001, 0.05, 0.5, 0.95])
    for du, hpi in [(-0.05, 1.5), (0.0, 1.0), (0.20, 0.4)]:
        out = macro_adjusted_pd(pd0, du, hpi)
        assert np.all(out > 0.0) and np.all(out < 1.0)


def test_pd_rises_with_unemployment_and_falls_with_house_prices():
    pd0 = np.array([0.01, 0.02])
    base = macro_adjusted_pd(pd0, 0.0, 1.0)
    assert np.all(macro_adjusted_pd(pd0, 0.03, 1.0) > base)
    assert np.all(macro_adjusted_pd(pd0, 0.0, 0.85) > base)
    assert np.all(macro_adjusted_pd(pd0, 0.0, 1.15) < base)


def test_neutral_macro_leaves_pd_unchanged():
    pd0 = np.array([0.003, 0.05, 0.4])
    assert np.allclose(macro_adjusted_pd(pd0, 0.0, 1.0), pd0)


def test_sensitivity_scales_the_shift():
    pd0 = np.array([0.01])
    low = macro_adjusted_pd(pd0, 0.03, 0.9, sensitivity=0.5)
    high = macro_adjusted_pd(pd0, 0.03, 0.9, sensitivity=2.0)
    assert high > low > pd0


# -------------------------------------------------------------- LGD behaviour
def test_lgd_within_bounds():
    ltv = np.linspace(0.30, 0.99, 40)
    for hpi in (0.5, 1.0, 1.4):
        lgd = haircut_lgd(ltv, hpi)
        assert np.all(lgd >= LGD_FLOOR) and np.all(lgd <= LGD_CAP)


def test_falling_house_prices_raise_lgd():
    ltv = np.array([0.80])
    assert haircut_lgd(ltv, 0.75) > haircut_lgd(ltv, 1.10)


def test_higher_ltv_raises_lgd():
    assert haircut_lgd(np.array([0.90]), 1.0) > haircut_lgd(np.array([0.60]), 1.0)


# -------------------------------------------------------------- EAD behaviour
def test_ead_amortises_and_never_goes_negative():
    bal = np.array([200_000.0, 200_000.0])
    term = np.array([10, 4])
    prev = amortising_ead(bal, term, 0)
    for year in range(1, 12):
        cur = amortising_ead(bal, term, year)
        assert np.all(cur >= 0.0)
        assert np.all(cur <= prev + 1e-9)
        prev = cur


# ------------------------------------------------------------------- staging
def test_stage_two_when_pd_triples():
    pd0 = np.array([0.01, 0.01, 0.01])
    current = np.array([0.02, 0.03, 0.05])  # ratios of 2, 3 and 5
    dpd = np.zeros(3)
    stage = assign_stage(current, pd0, dpd)
    assert stage[0] == 1                     # below the trigger
    assert stage[1] == 2 and stage[2] == 2   # at and above the trigger
    assert SICR_PD_RATIO == 3.0


def test_default_overrides_to_stage_three():
    stage = assign_stage(np.array([0.01]), np.array([0.01]),
                         np.array([DEFAULT_DPD]))
    assert stage[0] == 3


# ----------------------------------------------------------------------- ECL
def test_ecl_is_never_negative(portfolio, macro):
    for scenario in ("base", "upside", "downside"):
        out = scenario_ecl(portfolio, macro, scenario)
        assert (out["ecl_12m"] >= 0).all()
        assert (out["ecl_lifetime"] >= 0).all()
        assert (out["ecl_recognised"] >= 0).all()


def test_lifetime_ecl_exceeds_twelve_month(portfolio, macro):
    out = scenario_ecl(portfolio, macro, "base")
    assert (out["ecl_lifetime"] >= out["ecl_12m"] - 1e-9).all()


def test_stage_one_recognises_twelve_month_and_others_lifetime(portfolio, macro):
    out = scenario_ecl(portfolio, macro, "downside")
    s1 = out[out["stage"] == 1]
    rest = out[out["stage"] != 1]
    assert np.allclose(s1["ecl_recognised"], s1["ecl_12m"])
    if len(rest):
        assert np.allclose(rest["ecl_recognised"], rest["ecl_lifetime"])


def test_downside_is_more_expensive_than_base(portfolio, macro):
    base = scenario_ecl(portfolio, macro, "base")["ecl_recognised"].sum()
    down = scenario_ecl(portfolio, macro, "downside")["ecl_recognised"].sum()
    up = scenario_ecl(portfolio, macro, "upside")["ecl_recognised"].sum()
    assert down > base > up


def test_scenario_weights_sum_to_one():
    assert np.isclose(sum(SCENARIO_WEIGHTS.values()), 1.0)


def test_weighted_ecl_rejects_weights_that_do_not_sum_to_one(portfolio, macro):
    with pytest.raises(ValueError):
        weighted_ecl(portfolio, macro, weights={"base": 0.5, "downside": 0.2})


def test_weighted_total_lies_between_the_scenarios(portfolio, macro):
    _, summary = weighted_ecl(portfolio, macro)
    scenarios = summary[summary["scenario"] != "weighted"]
    weighted = float(summary.loc[summary["scenario"] == "weighted", "ecl_total"].iloc[0])
    assert scenarios["ecl_total"].min() <= weighted <= scenarios["ecl_total"].max()


def test_unknown_scenario_raises(portfolio, macro):
    with pytest.raises(ValueError):
        scenario_ecl(portfolio, macro, "no_such_scenario")
