"""End-to-end run: generates the data, runs all three engines, writes figures.

    python run_analysis.py

Prints every headline number quoted in the README and the slides, and writes
three figures to ``figures/``. Seeds are fixed throughout, so the output is
reproducible exactly.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data_generator import (annual_macro, generate_loan_portfolio,  # noqa: E402
                                generate_macro_scenarios)
from src.ifrs9_engine import weighted_ecl  # noqa: E402
from src.alm_hedging import nii_volatility, simulate_nii  # noqa: E402
from src.fraud_graph import (build_transaction_graph, graph_features,  # noqa: E402
                             train_mule_classifier)

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 200})
os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

SCENARIO_COLOURS = {"base": "tab:blue", "upside": "tab:green",
                    "downside": "tab:red"}

# --------------------------------------------------------------------- data
macro = generate_macro_scenarios()
loans = generate_loan_portfolio()
macro.to_csv("data/macro_scenarios.csv", index=False)
loans.to_csv("data/loan_portfolio.csv", index=False)
annual = annual_macro(macro)

print("=" * 78)
print("UK BANK RISK ENGINE")
print("=" * 78)
print(f"\nPortfolio: {len(loans):,} loans, gross balance "
      f"GBP {loans['balance'].sum()/1e9:.2f}bn, "
      f"mean LTV {loans['current_ltv'].mean():.1%}, "
      f"mean credit score {loans['credit_score'].mean():.0f}")

print("\n--- Macroeconomic scenarios, year 5 ---")
print(annual[annual["year"] == 5][
    ["scenario", "bank_rate", "unemployment", "hpi_index"]]
    .round(4).to_string(index=False))

# ------------------------------------------------------------------- IFRS 9
per_scenario, summary = weighted_ecl(loans, macro)
print("\n--- IFRS 9 expected credit loss ---")
show = summary.copy()
show["ecl_total_m"] = (show["ecl_total"] / 1e6).round(2)
show["coverage_bps"] = (show["coverage_ratio"] * 10_000).round(1)
print(show[["scenario", "weight", "ecl_total_m", "coverage_bps",
            "stage_1", "stage_2", "stage_3"]].to_string(index=False))

base_total = float(summary.loc[summary["scenario"] == "base", "ecl_total"].iloc[0])
down_total = float(summary.loc[summary["scenario"] == "downside", "ecl_total"].iloc[0])
print(f"\nDownside provision is {down_total/base_total:.1f}x the base case.")
down = per_scenario[per_scenario["scenario"] == "downside"]
print(f"Stage 2 migration in the downside: {(down['stage'] == 2).sum():,} loans "
      f"({(down['stage'] == 2).mean():.1%} of the book).")
s2 = down[down["stage"] == 2]
s1 = down[down["stage"] == 1]
print(f"Mean recognised ECL: Stage 1 GBP {s1['ecl_recognised'].mean():,.0f}, "
      f"Stage 2 GBP {s2['ecl_recognised'].mean():,.0f} "
      f"({s2['ecl_recognised'].mean()/s1['ecl_recognised'].mean():.1f}x).")

fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
order = ["upside", "base", "downside"]
totals = [float(summary.loc[summary["scenario"] == s, "ecl_total"].iloc[0]) / 1e6
          for s in order]
ax[0].bar(order, totals, color=[SCENARIO_COLOURS[s] for s in order], alpha=0.85)
weighted_total = float(summary.loc[summary["scenario"] == "weighted",
                                   "ecl_total"].iloc[0]) / 1e6
ax[0].axhline(weighted_total, ls="--", color="k", lw=1.0,
              label=f"probability weighted, £{weighted_total:.0f}m")
ax[0].set_ylabel("ECL provision (£m)")
ax[0].legend(frameon=False, fontsize=7.5)
stages = (per_scenario.groupby(["scenario", "stage"]).size()
          .unstack(fill_value=0).reindex(order))
bottom = np.zeros(len(order))
for stage, colour in [(1, "tab:blue"), (2, "tab:orange"), (3, "tab:red")]:
    vals = stages.get(stage, pd.Series(0, index=order)).to_numpy()
    ax[1].bar(order, vals, bottom=bottom, color=colour, alpha=0.85,
              label=f"Stage {stage}")
    bottom += vals
ax[1].set_ylabel("loans")
ax[1].legend(frameon=False, fontsize=7.5)
fig.tight_layout()
fig.savefig("figures/fig_ecl.pdf")

# ---------------------------------------------------------------------- ALM
print("\n--- ALM structural hedge ---")
vol = nii_volatility(annual)
print(vol.assign(unhedged_m=(vol["unhedged_range"] / 1e6).round(0),
                 hedged_m=(vol["hedged_range"] / 1e6).round(0))
      [["year", "unhedged_m", "hedged_m", "reduction_pct"]]
      .round(1).to_string(index=False))
print(f"\nHedge absorbs {vol['reduction_pct'].iloc[0]:.0f}% of the year 1 spread "
      f"and {vol['reduction_pct'].iloc[-1]:.0f}% by year 5, as the ladder rolls.")

fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
for s in order:
    r = simulate_nii(annual, s)
    ax[0].plot(r["year"], r["nii_unhedged"] / 1e6, "-o", ms=3,
               color=SCENARIO_COLOURS[s], lw=1.0, label=s)
    ax[1].plot(r["year"], r["nii_hedged"] / 1e6, "-o", ms=3,
               color=SCENARIO_COLOURS[s], lw=1.0, label=s)
lo = min(ax[0].get_ylim()[0], ax[1].get_ylim()[0])
hi = max(ax[0].get_ylim()[1], ax[1].get_ylim()[1])
for a, title in zip(ax, ["unhedged", "hedged"]):
    a.set_ylim(lo, hi)
    a.set_xlabel("year")
    a.set_title(title, fontsize=9)
    a.legend(frameon=False, fontsize=7.5)
ax[0].set_ylabel("net interest income (£m)")
fig.tight_layout()
fig.savefig("figures/fig_nii.pdf")

# -------------------------------------------------------------------- fraud
print("\n--- Graph money mule detection ---")
g, labels = build_transaction_graph()
feats = graph_features(g, labels)
result = train_mule_classifier(feats)
print(f"Network: {g.number_of_nodes():,} accounts, {g.number_of_edges():,} "
      f"payments, {sum(labels.values())} planted mules "
      f"({result['base_rate']:.1%} of accounts)")
print(f"ROC AUC {result['roc_auc']:.4f}, average precision "
      f"{result['average_precision']:.4f}")
print(f"Top {result['k']} alerts contain {result['hits_at_k']} of the "
      f"{result['positives_in_test']} mules in the test set: "
      f"precision {result['precision_at_k']:.2f} against a ceiling of "
      f"{result['precision_at_k_ceiling']:.2f}, recall "
      f"{result['recall_at_k']:.2f}")
print("\nThese scores measure whether the feature set recovers a structure that "
      "was planted by construction.\nThey are not an estimate of live "
      "performance and should not be read as one.")

means = feats.groupby("is_mule")[["conduit", "degree_ratio", "pagerank",
                                  "betweenness"]].mean()
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
ax[0].hist(feats.loc[feats["is_mule"] == 0, "conduit"], bins=40, density=True,
           alpha=0.6, color="tab:blue", label="ordinary")
ax[0].hist(feats.loc[feats["is_mule"] == 1, "conduit"], bins=40, density=True,
           alpha=0.6, color="tab:red", label="mule")
ax[0].set_xlabel("conduit score")
ax[0].set_ylabel("density")
ax[0].legend(frameon=False, fontsize=8)
labels_x = ["conduit", "degree ratio", "pagerank", "betweenness"]
x = np.arange(len(labels_x))
norm = means / means.max()
ax[1].bar(x - 0.19, norm.loc[0], 0.38, color="tab:blue", alpha=0.8, label="ordinary")
ax[1].bar(x + 0.19, norm.loc[1], 0.38, color="tab:red", alpha=0.8, label="mule")
ax[1].set_xticks(x)
ax[1].set_xticklabels(labels_x, fontsize=7.5)
ax[1].set_ylabel("mean, scaled to the larger group")
ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("figures/fig_fraud.pdf")

print("\nFigures written to figures/. Done.")
