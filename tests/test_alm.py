"""Unit tests for the ALM structural hedge and the graph fraud engine."""
import numpy as np
import pytest

from src.data_generator import annual_macro, generate_macro_scenarios
from src.alm_hedging import (deposit_balance, deposit_rate_from_beta,
                             nii_volatility, simulate_nii, swap_ladder)
from src.fraud_graph import (build_transaction_graph, graph_features,
                             train_mule_classifier)


@pytest.fixture(scope="module")
def macro():
    return annual_macro(generate_macro_scenarios(seed=2))


# ------------------------------------------------------------------- deposits
def test_deposits_decay_and_never_grow():
    years = np.arange(0, 6)
    rate = np.full(6, 0.05)
    dep = deposit_rate_from_beta(rate)
    bal = deposit_balance(100.0, years, rate, dep)
    assert np.isclose(bal[0], 100.0)
    assert np.all(np.diff(bal) < 0)
    assert np.all(bal > 0)


def test_a_wider_rate_gap_accelerates_runoff():
    years = np.arange(0, 6)
    slow = deposit_balance(100.0, years, np.full(6, 0.02), np.full(6, 0.02))
    fast = deposit_balance(100.0, years, np.full(6, 0.08), np.full(6, 0.01))
    assert fast[-1] < slow[-1]


def test_deposit_beta_bounds_the_pass_through():
    market = np.array([0.00, 0.02, 0.06])
    dep = deposit_rate_from_beta(market, beta=0.35)
    assert np.all(dep >= 0.0)
    assert np.all(dep <= market + 1e-12)


# ---------------------------------------------------------------- swap ladder
def test_ladder_receives_a_moving_average_of_past_rates():
    market = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
    ladder = swap_ladder(100.0, 5, market, historical_rate=0.05)
    # Flat rates at the historical level mean the ladder is exactly neutral.
    assert np.allclose(ladder["swap_net"], 0.0, atol=1e-9)


def test_receiver_ladder_gains_when_rates_fall():
    falling = np.array([0.04, 0.03, 0.02, 0.01, 0.01])
    ladder = swap_ladder(100.0, 5, falling, historical_rate=0.05)
    assert np.all(ladder["swap_net"] > 0)


def test_receiver_ladder_loses_when_rates_rise():
    rising = np.array([0.06, 0.07, 0.08, 0.09, 0.09])
    ladder = swap_ladder(100.0, 5, rising, historical_rate=0.03)
    assert np.all(ladder["swap_net"] < 0)


def test_ladder_notional_is_conserved():
    market = np.array([0.03, 0.05, 0.04])
    ladder = swap_ladder(500.0, 5, market, historical_rate=0.03)
    assert np.allclose(ladder["swap_pay"], 500.0 * market)


# ----------------------------------------------------------------------- NII
def test_hedge_narrows_the_range_of_income(macro):
    vol = nii_volatility(macro)
    assert (vol["hedged_range"] < vol["unhedged_range"]).all()
    assert vol["reduction_pct"].iloc[0] > 20.0


def test_hedge_protection_decays_as_the_ladder_rolls(macro):
    vol = nii_volatility(macro)
    # Old tranches roll off and are replaced at prevailing rates, so the
    # proportion of the swing that is absorbed falls over the horizon.
    assert vol["reduction_pct"].iloc[0] > vol["reduction_pct"].iloc[-1]


def test_hedge_loses_money_when_rates_rise(macro):
    down = simulate_nii(macro, "downside")  # the high rate path
    assert down["swap_net"].mean() < 0
    assert (down["nii_hedged"] < down["nii_unhedged"]).all()


def test_zero_hedge_ratio_is_a_no_op(macro):
    out = simulate_nii(macro, "base", hedge_ratio=0.0)
    assert np.allclose(out["nii_hedged"], out["nii_unhedged"])


def test_unknown_scenario_raises(macro):
    with pytest.raises(ValueError):
        simulate_nii(macro, "no_such_scenario")


# ------------------------------------------------------------- fraud detection
@pytest.fixture(scope="module")
def network():
    g, labels = build_transaction_graph(n_accounts=800, n_rings=10, seed=3)
    return g, labels


def test_graph_is_built_with_the_expected_number_of_mules(network):
    g, labels = network
    assert g.number_of_nodes() == 800
    assert sum(labels.values()) == 60
    assert g.number_of_edges() > 800


def test_features_are_finite_and_well_scaled(network):
    g, labels = network
    feats = graph_features(g, labels)
    assert len(feats) == g.number_of_nodes()
    assert np.isfinite(feats.select_dtypes("number").to_numpy()).all()
    assert feats["conduit"].between(0.0, 1.0).all()
    assert (feats["in_degree"] >= 0).all() and (feats["out_degree"] >= 0).all()


def test_mules_look_more_like_conduits_than_ordinary_accounts(network):
    g, labels = network
    feats = graph_features(g, labels)
    mules = feats[feats["is_mule"] == 1]
    rest = feats[feats["is_mule"] == 0]
    assert mules["conduit"].mean() > rest["conduit"].mean()
    assert mules["betweenness"].mean() > rest["betweenness"].mean()


def test_classifier_beats_a_coin_toss(network):
    g, labels = network
    result = train_mule_classifier(graph_features(g, labels), seed=3)
    assert result["roc_auc"] > 0.75
    assert result["average_precision"] > result["base_rate"]
    assert 0.0 <= result["precision_at_k"] <= result["precision_at_k_ceiling"] + 1e-12
