# UK Bank Risk Engine

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![Tests](https://img.shields.io/badge/tests-34%20passing-brightgreen)
![Licence](https://img.shields.io/badge/licence-MIT-lightgrey)

Three risk engines that sit on a UK retail bank's balance sheet, implemented end
to end on synthetic data: IFRS 9 expected credit loss under macroeconomic stress,
an asset and liability structural hedge, and graph-based money mule detection.

Everything is reproducible. Seeds are fixed, so `python run_analysis.py`
regenerates every number below exactly.

> **All data here is synthetic.** No figure is calibrated to a real bank, a real
> portfolio or a real payment network. The purpose is to implement the mechanics
> correctly and show what they do, not to estimate anything.

## Headline results

**IFRS 9 expected credit loss** on a synthetic book of 10,000 mortgages, gross
balance £2.80bn.

| Scenario | Weight | Provision | Coverage | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|---|---|---|
| Upside | 0.20 | £1.99m | 7 bps | 9,923 | 0 | 77 |
| Base | 0.50 | £2.04m | 7 bps | 9,923 | 0 | 77 |
| Downside | 0.30 | £53.72m | 192 bps | 6,932 | 2,991 | 77 |
| **Weighted** | 1.00 | **£17.53m** | **63 bps** | | | |

The severe downside raises the provision by a factor of **26**, and almost all of
that comes from stage migration rather than from higher loss rates. 2,991 loans,
just under 30% of the book, breach the SICR test and move from a twelve-month to
a lifetime loss allowance. Mean recognised ECL rises from £228 in Stage 1 to
£17,224 in Stage 2, a factor of 76. This cliff is the single most important
feature of IFRS 9 and the reason provisions move so violently in a downturn.

**ALM structural hedge**, a mature five-year SONIA receiver ladder on 80% of the
rate-insensitive deposit base.

| Year | Unhedged NII range | Hedged range | Absorbed |
|---|---|---|---|
| 1 | £438m | £131m | 70% |
| 2 | £975m | £378m | 61% |
| 3 | £1,318m | £650m | 51% |
| 4 | £1,339m | £855m | 36% |
| 5 | £1,406m | £1,073m | 24% |

The protection **decays as the ladder rolls**. Old tranches mature and are
replaced at prevailing rates, so a five-year ladder defends the near term
strongly and converges towards the unhedged position over its own tenor. A hedge
is a delay, not a cure.

**Graph money mule detection** on a synthetic network of 3,000 accounts and
7,831 payments containing 25 planted rings.

| Metric | Value |
|---|---|
| Mule base rate | 5.0% |
| ROC AUC | 0.997 |
| Average precision | 0.941 |
| Recall in the top 100 alerts | 100% (45 of 45) |

**These scores are not evidence that the method works on real data.** The rings
were planted with a distinctive topology, so what is being measured is whether
the feature set recovers a structure that was put there by construction. A real
mule network is adversarial, sparser and far less tidy. The honest claim is that
the pipeline is correct, not that it is accurate.

## Repository layout

```
uk-bank-risk-engine/
├── .github/workflows/python-app.yml   Lint and test on 3.10, 3.11 and 3.12
├── src/
│   ├── data_generator.py              Synthetic macro scenarios and loan book
│   ├── ifrs9_engine.py                PD, LGD, EAD, SICR staging, ECL
│   ├── alm_hedging.py                 Deposit decay and the swap ladder
│   └── fraud_graph.py                 Graph features and the mule classifier
├── tests/
│   ├── test_ifrs9.py                  20 tests
│   └── test_alm.py                    14 tests
├── app/dashboard.py                   Streamlit dashboard, three tabs
├── run_analysis.py                    Full pipeline, prints results, writes figures
├── Dockerfile
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
python run_analysis.py        # results and figures
python -m pytest tests -q     # 34 tests
streamlit run app/dashboard.py
```

Or with Docker:

```bash
docker build -t uk-bank-risk-engine .
docker run -p 8501:8501 uk-bank-risk-engine
```

## The models

### IFRS 9 expected credit loss

Lifetime ECL discounted at the effective interest rate:

```
ECL = Σ_t  S_{t-1} · PD_t · LGD_t · EAD_t / (1 + r)^t
```

where `S_{t-1}` is the probability of surviving to the start of year `t`, so
that marginal rather than cumulative default is used in each period.

**Macro-adjusted PD** through a logistic link:

```
logit(PD_t) = logit(PD_0) + s_i · ( β₁ · ΔUnemployment_t − β₂ · (HPI_t − 1) )
```

The loan-level multiplier `s_i` rises with LTV and with the debt service ratio.
Without it every loan receives the same shift in log-odds, the PD ratio used by
the SICR test is then almost identical across the book, and staging becomes an
all-or-nothing switch for the whole portfolio. That is an artefact, not a result.

**LGD** from collateral value net of a forced sale haircut:

```
LGD_t = clip( 1 − (1 − haircut) · HPI_t / LTV_t ,  0.05,  0.95 )
```

**SICR staging.** Stage 3 at 90 or more days past due. Stage 2 when lifetime PD
has risen by a factor of 3 or more since origination. Stage 1 otherwise, carrying
a twelve-month allowance rather than a lifetime one.

### ALM structural hedge

Behavioural deposit runoff:

```
D(t) = D₀ · exp(−λt),    λ = λ_base + γ · max(r_market − r_deposit, 0)
```

The hedge is a rolling ladder of receiver swaps, one tranche maturing and being
replaced each year, receiving a moving average of past fixed rates and paying
today's floating rate. That averaging is the entire mechanism.

### Graph mule detection

A directed payment graph with in and out degree, a degree ratio, flow volumes, a
conduit score, PageRank and betweenness centrality, fed to a gradient boosted
classifier. The conduit score

```
conduit = min(inflow, outflow) / (max(inflow, outflow) + 1)
```

is close to 1 for an account that passes on almost exactly what it receives,
which is what a layering account does. It is defined symmetrically so that
accounts with only inflows or only outflows do not dominate through a near-zero
denominator.

## Deviations from the original project plan

Four changes were made deliberately, each because the original specification was
either wrong or ambiguous.

1. **LGD sign.** The plan writes the haircut term as `LTV · (1 − ΔHPI)`. That has
   rising house prices *increasing* loss given default. Corrected to a cumulative
   house price index applied to collateral value.
2. **SICR threshold.** "ΔPD ≥ 200%" is ambiguous between a doubling and a
   tripling. Implemented as a PD ratio of 3.0, which matches the plan's own test
   case that staging should shift when PD triples.
3. **Hedge notional.** The hedge is sized on the rate-insensitive share of
   deposits, `deposits × (1 − beta)`, not the full deposit base, and the ladder
   is mature at the start rather than built from nothing. A ladder built from
   zero has almost no effect in its first years, which flatters the unhedged
   comparison.
4. **Classifier.** `HistGradientBoostingClassifier` from scikit-learn replaces
   XGBoost. It is the same class of model with one fewer dependency, and the
   interface is unchanged if you prefer to swap it back.

## Limitations

- Everything is synthetic, so no result is evidence about any real bank.
- The three engines share a data generator but are not jointly calibrated. A
  real balance sheet would link credit losses, deposit behaviour and rates.
- The macro model is a deterministic ramp with noise, not a stochastic scenario
  generator, so tail risk is imposed rather than emergent.
- Basel capital treatment, ring-fencing constraints and the regulatory
  reimbursement rules referenced in the project brief are not modelled.

## Author

Nihal Naresh Kumar, MSc Mathematical Finance

## Licence

MIT.
