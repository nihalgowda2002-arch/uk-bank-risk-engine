"""Graph features and a classifier for money mule detection.

Builds a synthetic directed payment network containing a background of ordinary
accounts and a number of planted mule rings, computes topological features for
each account, and trains a classifier on those features.

The plan specifies XGBoost. This module uses scikit-learn's
``HistGradientBoostingClassifier``, which is the same class of model and removes
a dependency. The interface is unchanged if you wish to substitute XGBoost.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


def build_transaction_graph(n_accounts: int = 3000, n_rings: int = 25,
                            ring_size: int = 6, seed: int = 20260818
                            ) -> tuple[nx.DiGraph, dict[str, int]]:
    """A payment network with planted mule rings.

    Ordinary accounts transact with a small number of counterparties. A mule
    ring collects funds from several victim accounts into an entry node, passes
    them through layering accounts in rapid succession, and disperses them from
    an exit node. The distinguishing feature is the throughput, not the volume.
    """
    rng = np.random.default_rng(seed)
    g = nx.DiGraph()
    accounts = [f"A{i:05d}" for i in range(n_accounts)]
    g.add_nodes_from(accounts)
    labels = {a: 0 for a in accounts}

    # Background activity: each account sends to a handful of counterparties.
    for a in accounts:
        for _ in range(rng.integers(1, 5)):
            b = accounts[rng.integers(0, n_accounts)]
            if a == b:
                continue
            amount = float(rng.lognormal(np.log(250), 1.1))
            if g.has_edge(a, b):
                g[a][b]["amount"] += amount
                g[a][b]["count"] += 1
            else:
                g.add_edge(a, b, amount=amount, count=1)

    # Planted rings.
    pool = list(rng.choice(accounts, size=n_rings * ring_size, replace=False))
    for r in range(n_rings):
        ring = pool[r * ring_size:(r + 1) * ring_size]
        for m in ring:
            labels[m] = 1
        victims = list(rng.choice(accounts, size=rng.integers(4, 10), replace=False))
        entry, exit_ = ring[0], ring[-1]
        for v in victims:
            amt = float(rng.lognormal(np.log(4000), 0.6))
            g.add_edge(v, entry, amount=amt, count=1)
        for i in range(len(ring) - 1):
            amt = float(rng.lognormal(np.log(3500), 0.4))
            g.add_edge(ring[i], ring[i + 1], amount=amt, count=int(rng.integers(3, 9)))
        for _ in range(rng.integers(3, 8)):
            d = accounts[rng.integers(0, n_accounts)]
            if d != exit_:
                g.add_edge(exit_, d, amount=float(rng.lognormal(np.log(3000), 0.5)),
                           count=1)

    return g, labels


def graph_features(g: nx.DiGraph, labels: dict[str, int] | None = None) -> pd.DataFrame:
    """Topological features for every account in the network."""
    pagerank = nx.pagerank(g, weight="amount")
    betweenness = nx.betweenness_centrality(g, k=min(400, g.number_of_nodes()),
                                            seed=7)
    in_amt = {n: 0.0 for n in g}
    out_amt = {n: 0.0 for n in g}
    for u, v, d in g.edges(data=True):
        out_amt[u] += d["amount"]
        in_amt[v] += d["amount"]

    rows = []
    for n in g.nodes:
        din, dout = g.in_degree(n), g.out_degree(n)
        inflow, outflow = in_amt[n], out_amt[n]
        rows.append({
            "account": n,
            "in_degree": din,
            "out_degree": dout,
            "degree_ratio": (din + 1.0) / (dout + 1.0),
            "in_amount": inflow,
            "out_amount": outflow,
            # Conduit score: close to 1 when an account passes on almost
            # exactly what it receives, which is what a layering account does.
            # Defined symmetrically so that accounts with only inflows or only
            # outflows do not dominate through a division by a near-zero
            # denominator.
            "conduit": min(inflow, outflow) / (max(inflow, outflow) + 1.0),
            "net_flow": inflow - outflow,
            "pagerank": pagerank[n],
            "betweenness": betweenness[n],
        })
    out = pd.DataFrame(rows)
    if labels is not None:
        out["is_mule"] = out["account"].map(labels).astype(int)
    return out


def train_mule_classifier(features: pd.DataFrame, seed: int = 20260818) -> dict:
    """Train and evaluate the classifier on a held-out split."""
    cols = ["in_degree", "out_degree", "degree_ratio", "in_amount", "out_amount",
            "conduit", "net_flow", "pagerank", "betweenness"]
    x = features[cols].to_numpy()
    y = features["is_mule"].to_numpy()

    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.30, random_state=seed, stratify=y)

    model = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.08, max_depth=5, random_state=seed)
    model.fit(x_tr, y_tr)
    p = model.predict_proba(x_te)[:, 1]

    # An investigations team can work only a fixed number of alerts a day, so
    # what matters is how many true mules appear in the top of the queue.
    # Precision at 100 is capped by the number of mules present, so the ceiling
    # is reported alongside it to keep the figure honest.
    k = min(100, len(y_te))
    order = np.argsort(-p)
    hits = int(y_te[order[:k]].sum())
    n_positive = int(y_te.sum())

    return {
        "model": model,
        "features": cols,
        "n_train": len(y_tr),
        "n_test": len(y_te),
        "base_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y_te, p)),
        "average_precision": float(average_precision_score(y_te, p)),
        "k": k,
        "hits_at_k": hits,
        "positives_in_test": n_positive,
        "precision_at_k": hits / k,
        "precision_at_k_ceiling": min(n_positive, k) / k,
        "recall_at_k": hits / n_positive if n_positive else float("nan"),
    }
