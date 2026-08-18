"""Streamlit dashboard for the UK bank risk engine.

Run with::

    streamlit run app/dashboard.py

Three tabs: IFRS 9 stress testing, the ALM treasury desk, and the graph fraud
engine. Every figure is computed live from the modules in ``src``.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generator import (annual_macro, generate_loan_portfolio,  # noqa: E402
                                generate_macro_scenarios)
from src.ifrs9_engine import scenario_ecl  # noqa: E402
from src.alm_hedging import nii_volatility, simulate_nii  # noqa: E402
from src.fraud_graph import (build_transaction_graph, graph_features,  # noqa: E402
                             train_mule_classifier)

st.set_page_config(page_title="UK Bank Risk Engine", layout="wide")

SCENARIO_COLOURS = {"base": "#1f77b4", "upside": "#2ca02c", "downside": "#d62728"}


@st.cache_data
def load_book(n_loans: int = 10_000):
    macro = generate_macro_scenarios()
    loans = generate_loan_portfolio(n=n_loans)
    return macro, annual_macro(macro), loans


@st.cache_data
def load_network(n_accounts: int = 3000, n_rings: int = 25):
    graph, labels = build_transaction_graph(n_accounts=n_accounts, n_rings=n_rings)
    feats = graph_features(graph, labels)
    return feats


macro, annual, loans = load_book()

st.title("UK Bank Risk Engine")
st.caption("Synthetic data throughout. Nothing here is calibrated to a real "
           "balance sheet, and no figure should be read as an estimate.")

tab_ecl, tab_alm, tab_fraud = st.tabs(
    ["IFRS 9 stress testing", "ALM treasury desk", "Graph fraud engine"])

# ------------------------------------------------------------------ IFRS 9
with tab_ecl:
    st.subheader("Expected credit loss under macroeconomic stress")
    left, right = st.columns([1, 3])

    with left:
        scenario = st.selectbox("Scenario", ["base", "upside", "downside"], index=0)
        unemp_shift = st.slider("Additional unemployment shock (pp)",
                                0.0, 5.0, 0.0, 0.25) / 100.0
        hpi_shift = st.slider("Additional house price shock (%)",
                              -30.0, 10.0, 0.0, 1.0) / 100.0

    stressed = annual.copy()
    mask = stressed["scenario"] == scenario
    stressed.loc[mask, "d_unemployment"] += unemp_shift
    stressed.loc[mask, "hpi_index"] *= (1.0 + hpi_shift)

    result = scenario_ecl(loans, stressed, scenario)
    provision = result["ecl_recognised"].sum()
    coverage = provision / loans["balance"].sum() * 10_000

    with right:
        c1, c2, c3 = st.columns(3)
        c1.metric("Provision", f"£{provision/1e6:,.1f}m")
        c2.metric("Coverage", f"{coverage:,.0f} bps")
        c3.metric("Stage 2 share", f"{(result['stage'] == 2).mean():.1%}")

        counts = result["stage"].value_counts().reindex([1, 2, 3], fill_value=0)
        fig = go.Figure(go.Bar(
            x=[f"Stage {i}" for i in counts.index],
            y=counts.to_numpy(),
            marker_color=["#1f77b4", "#ff7f0e", "#d62728"]))
        fig.update_layout(height=300, yaxis_title="loans",
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Provision by stage**")
    by_stage = (result.groupby("stage")
                .agg(loans=("loan_id", "count"),
                     provision=("ecl_recognised", "sum"),
                     mean_provision=("ecl_recognised", "mean"))
                .reset_index())
    st.dataframe(by_stage.style.format(
        {"provision": "£{:,.0f}", "mean_provision": "£{:,.0f}"}),
        use_container_width=True)

# --------------------------------------------------------------------- ALM
with tab_alm:
    st.subheader("Net interest income and the structural hedge")
    left, right = st.columns([1, 3])

    with left:
        hedge_ratio = st.slider("Hedge ratio", 0.0, 1.0, 0.80, 0.05)
        tenor = st.slider("Ladder tenor (years)", 2, 10, 5, 1)
        beta = st.slider("Deposit beta", 0.0, 1.0, 0.35, 0.05)

    frames = [simulate_nii(annual, s, hedge_ratio=hedge_ratio, tenor=tenor,
                           deposit_beta=beta)
              for s in ("base", "upside", "downside")]
    allof = pd.concat(frames, ignore_index=True)

    with right:
        fig = go.Figure()
        for s in ("base", "upside", "downside"):
            sub = allof[allof["scenario"] == s]
            fig.add_trace(go.Scatter(x=sub["year"], y=sub["nii_unhedged"] / 1e6,
                                     name=f"{s}, unhedged", line=dict(
                                         color=SCENARIO_COLOURS[s], dash="dot")))
            fig.add_trace(go.Scatter(x=sub["year"], y=sub["nii_hedged"] / 1e6,
                                     name=f"{s}, hedged", line=dict(
                                         color=SCENARIO_COLOURS[s])))
        fig.update_layout(height=380, yaxis_title="net interest income (£m)",
                          xaxis_title="year", margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    vol = nii_volatility(annual, hedge_ratio=hedge_ratio, tenor=tenor,
                         deposit_beta=beta)
    c1, c2 = st.columns(2)
    c1.metric("Year 1 spread absorbed", f"{vol['reduction_pct'].iloc[0]:.0f}%")
    c2.metric("Year 5 spread absorbed", f"{vol['reduction_pct'].iloc[-1]:.0f}%")
    st.caption("The hedge loses money when rates rise. That is the intended "
               "behaviour: it is bought to narrow the range of outcomes, not "
               "to raise income.")

# ------------------------------------------------------------------- fraud
with tab_fraud:
    st.subheader("Money mule detection on the payment graph")
    st.warning("The rings in this network were planted by construction, so the "
               "scores below measure whether the features recover a known "
               "structure. They are not an estimate of live performance.")

    feats = load_network()
    result = train_mule_classifier(feats)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accounts", f"{len(feats):,}")
    c2.metric("Mule base rate", f"{result['base_rate']:.1%}")
    c3.metric("ROC AUC", f"{result['roc_auc']:.3f}")
    c4.metric(f"Recall in top {result['k']}", f"{result['recall_at_k']:.0%}")

    feature = st.selectbox("Feature", ["conduit", "betweenness", "pagerank",
                                       "degree_ratio", "in_degree", "out_degree"])
    plot_df = feats.assign(label=np.where(feats["is_mule"] == 1, "mule", "ordinary"))
    fig = px.histogram(plot_df, x=feature, color="label", barmode="overlay",
                       nbins=60, histnorm="probability density",
                       color_discrete_map={"mule": "#d62728",
                                           "ordinary": "#1f77b4"})
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Highest scoring accounts**")
    x = feats[result["features"]].to_numpy()
    feats = feats.assign(score=result["model"].predict_proba(x)[:, 1])
    st.dataframe(
        feats.nlargest(15, "score")[["account", "score", "conduit",
                                     "betweenness", "pagerank", "is_mule"]]
        .style.format({"score": "{:.3f}", "conduit": "{:.3f}",
                       "betweenness": "{:.4f}", "pagerank": "{:.5f}"}),
        use_container_width=True)
