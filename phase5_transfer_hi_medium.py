"""Phase 5 -- FROZEN transfer test on HI-Medium: the scale rung.

31,898,238 rows (6.3x HI-Small), tx-level positive 0.1104% (HI-family).
Same frozen recipe as every transfer run. Memory feasibility on a 32 GB
machine is itself part of the experiment -- if it dies, that is the finding.

PLAN.md's own named bite: "Synthetic data ... a method that works here may be
learning the generator. Say so." Every method so far was developed and scored
on HI-Small only. This script runs the operational-best recipe, FROZEN, on a
dataset from a different generation run.

The data: TRUE LI-Small from IBM's Kaggle archive (LI-Small_Trans.csv, 6,924,049
rows -- the HF mirror tested earlier was exactly a 50% subsample). Same
11-column schema, Is Laundering present, tx-level positive 0.0515%.
This run settles the phase3 confound: mirror said AUROC 0.6901 -- was that
generator-overfit or a starved half-graph?

FROZEN RECIPE (decided before this script was run, zero LI-tuning):
  - features: the 9 counting/cycle features + hub-excluded damped CF + binary
    has-any-cross-currency flag (the operational best from phase2_invariants.py)
  - hub rule: exclude counterparties with popularity > 100 (the constant from
    HI-Small's empty 94-566 gap; LI's own distribution is printed for context
    but the constant is NOT re-derived)
  - protocol: 60/40 stratified split from sorted accounts, seeds 0-4,
    AUROC + shuffle control + distinct count + p@100/500/1000/npos
  - plus the triage-tier readout: positive rate of the cross-currency tier
    within each seed's top-1000 worklist (was 96.7-98.2% on HI-Small)

No try/except. If something breaks, it raises.
"""
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FILES = [os.path.join(DATA, "hi_medium_full.parquet")]

SEEDS = [0, 1, 2, 3, 4]
K_VALUES = [100, 500, 1000]
HUB_CUTOFF = 100  # FROZEN from HI-Small; not re-derived here
WORKLIST_K = 1000

print("=" * 70)
print("LOAD (HI-Medium, full graph)")
tx = pd.concat([pd.read_parquet(f) for f in FILES], ignore_index=True)
tx["from_acct"] = tx["From Bank"].astype(str) + ":" + tx["Account"]
tx["to_acct"] = tx["To Bank"].astype(str) + ":" + tx["Account.1"]
print(f"  combined: {len(tx):,} rows")

all_accts = sorted(set(tx["from_acct"]) | set(tx["to_acct"]))
n = len(all_accts)
idx_of = {a: i for i, a in enumerate(all_accts)}
acct_index = pd.Index(all_accts)

flagged = tx[tx["Is Laundering"] == 1]
pos_accts = set(flagged["from_acct"]) | set(flagged["to_acct"])
label = acct_index.isin(pos_accts).astype(int)
n_pos = int(label.sum())
print(f"  accounts: {n:,}   positive: {n_pos:,} = {n_pos/n:.4%}")

print()
print("=" * 70)
print("FEATURES (identical build to the HI-Small scripts)")
out_amt = tx.groupby("from_acct")["Amount Paid"].sum()
in_amt = tx.groupby("to_acct")["Amount Received"].sum()
out_cnt = tx.groupby("from_acct").size()
in_cnt = tx.groupby("to_acct").size()
in_degree = tx.groupby("to_acct")["from_acct"].nunique()
out_degree = tx.groupby("from_acct")["to_acct"].nunique()
banks_long = pd.concat([
    tx[["from_acct", "From Bank"]].rename(columns={"from_acct": "acct", "From Bank": "bank"}),
    tx[["from_acct", "To Bank"]].rename(columns={"from_acct": "acct", "To Bank": "bank"}),
    tx[["to_acct", "From Bank"]].rename(columns={"to_acct": "acct", "From Bank": "bank"}),
    tx[["to_acct", "To Bank"]].rename(columns={"to_acct": "acct", "To Bank": "bank"}),
], ignore_index=True)
distinct_banks = banks_long.groupby("acct")["bank"].nunique()
edges = tx.loc[tx["from_acct"] != tx["to_acct"], ["from_acct", "to_acct"]].drop_duplicates()
reciprocal = edges.merge(edges, left_on=["from_acct", "to_acct"],
                          right_on=["to_acct", "from_acct"])
reciprocal_count = reciprocal.groupby("from_acct_x")["to_acct_x"].nunique()
edges2 = edges.rename(columns={"from_acct": "u", "to_acct": "v"})
hop2 = edges2.merge(edges2, left_on="v", right_on="u", suffixes=("1", "2"))
hop2 = hop2.rename(columns={"u1": "u", "v2": "w"})[["u", "v1", "w"]]
hop2 = hop2[hop2["u"] != hop2["w"]]
triangles = hop2.merge(edges2, left_on=["w", "u"], right_on=["u", "v"], suffixes=("", "_close"))
triangles = triangles[["u", "v1", "w"]].drop_duplicates().rename(columns={"v1": "v"})
tri_long = pd.concat([
    triangles[["u"]].rename(columns={"u": "acct"}),
    triangles[["v"]].rename(columns={"v": "acct"}),
    triangles[["w"]].rename(columns={"w": "acct"}),
], ignore_index=True)
tri3_count = tri_long.groupby("acct").size()

acc = pd.DataFrame(index=all_accts)
acc.index.name = "acct"
acc["degree"] = (in_degree.reindex(acc.index).fillna(0) + out_degree.reindex(acc.index).fillna(0))
acc["in_degree"] = in_degree.reindex(acc.index).fillna(0)
acc["out_degree"] = out_degree.reindex(acc.index).fillna(0)
acc["amount_out"] = out_amt.reindex(acc.index).fillna(0.0)
acc["amount_in"] = in_amt.reindex(acc.index).fillna(0.0)
acc["num_transactions"] = (out_cnt.reindex(acc.index).fillna(0) + in_cnt.reindex(acc.index).fillna(0))
acc["distinct_banks"] = distinct_banks.reindex(acc.index).fillna(0)
acc["reciprocal_count"] = reciprocal_count.reindex(acc.index).fillna(0)
acc["tri3_count"] = tri3_count.reindex(acc.index).fillna(0)
X_static = acc[["degree", "in_degree", "out_degree", "amount_out", "amount_in",
                "num_transactions", "distinct_banks", "reciprocal_count", "tri3_count"]].to_numpy()

counterparty_long = pd.concat([
    tx[["from_acct", "to_acct"]].rename(columns={"from_acct": "acct", "to_acct": "counterparty"}),
    tx[["to_acct", "from_acct"]].rename(columns={"to_acct": "acct", "from_acct": "counterparty"}),
], ignore_index=True).drop_duplicates()
popularity = counterparty_long.groupby("counterparty")["acct"].nunique()
hubs = popularity[popularity > HUB_CUTOFF]
print(f"  counterparty popularity on LI: median {popularity.median():.0f}, "
      f"max {popularity.max():,}")
print(f"  frozen hub rule (>{HUB_CUTOFF}) excludes {len(hubs)} counterparties "
      f"(top: {dict(hubs.sort_values(ascending=False).head(5))})")
kept_cp = counterparty_long[~counterparty_long["counterparty"].isin(set(hubs.index))]
pop_kept = kept_cp.groupby("counterparty")["acct"].nunique()
idf = 1.0 / np.sqrt(1.0 + pop_kept)
rows_ = kept_cp["acct"].map(idx_of).to_numpy()
cols_ = kept_cp["counterparty"].map(idx_of).to_numpy()
M = sp.csr_matrix((idf.reindex(kept_cp["counterparty"]).to_numpy(), (rows_, cols_)), shape=(n, n))
row_norms = np.sqrt(M.multiply(M).sum(axis=1)).A.flatten()
row_norms[row_norms == 0] = 1.0
M = sp.diags(1.0 / row_norms) @ M

cross = tx.loc[tx["Payment Currency"] != tx["Receiving Currency"]]
fx_accts = set(cross["from_acct"]) | set(cross["to_acct"])
has_fx = acct_index.isin(fx_accts).astype(float)
print(f"  cross-currency rows: {len(cross):,} ({len(cross)/len(tx):.4%}); "
      f"accounts with any: {int(has_fx.sum()):,} ({has_fx.mean():.2%})")


def precision_at_k(scores, labels, k):
    """Top-k by score. Ties at the cutoff are included, so sel may exceed k."""
    threshold = np.partition(scores, -k)[-k]
    selected = scores >= threshold
    return int(labels[selected].sum()), int(selected.sum())


idxs = np.arange(n)
results = []
print()
print("=" * 70)
print("5-SEED RUNS -- frozen recipe, LI data")
for seed in SEEDS:
    tr, te = train_test_split(idxs, test_size=0.4, stratify=label, random_state=seed)
    ytr, yte = label[tr], label[te]
    npt = int(yte.sum())
    rng = np.random.default_rng(seed)
    ysh = yte.copy()
    rng.shuffle(ysh)

    u1 = M[tr].T @ ytr
    h1 = np.asarray(M @ u1).flatten()
    X = np.column_stack([X_static, h1, has_fx])
    sc = StandardScaler()
    mdl = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    mdl.fit(sc.fit_transform(X[tr]), ytr)
    s = mdl.predict_proba(sc.transform(X[te]))[:, 1]

    row = {"seed": seed, "auroc": roc_auc_score(yte, s),
           "auroc_shuf": roc_auc_score(ysh, s),
           "distinct": int(pd.Series(s).nunique()), "n_pos_test": npt}
    for k in K_VALUES + [npt]:
        key = k if k in K_VALUES else "npos"
        hits, sel = precision_at_k(s, yte, k)
        row[f"p@{key}"] = hits / sel
        row[f"sel@{key}"] = sel

    top = np.argsort(-s)[:WORKLIST_K]
    top_accts = acct_index[te[top]]
    top_pos = yte[top]
    in_fx = top_accts.isin(fx_accts)
    n_fx = int(in_fx.sum())
    fx_pos = int(top_pos[in_fx].sum())
    row["fx_tier_n"] = n_fx
    row["fx_tier_rate"] = fx_pos / n_fx if n_fx else float("nan")
    rest = WORKLIST_K - n_fx
    row["rest_rate"] = int(top_pos[~in_fx].sum()) / rest if rest else float("nan")
    results.append(row)
    print(f"  seed {seed}: AUROC {row['auroc']:.4f} (shuf {row['auroc_shuf']:.4f})  "
          f"distinct {row['distinct']:,}  p@100 {row['p@100']:.4f} (sel {row['sel@100']})  "
          f"p@500 {row['p@500']:.4f}  p@1000 {row['p@1000']:.4f}  "
          f"p@npos({npt}) {row['p@npos']:.4f}")
    print(f"          fx tier {fx_pos}/{n_fx} = {row['fx_tier_rate']:.1%}  |  "
          f"rest of worklist {row['rest_rate']:.1%}")

res = pd.DataFrame(results)
base = n_pos / n
print()
print("=" * 70)
print("TRANSFER VERDICT (mean across 5 seeds) vs the HI-Small numbers")
print(f"  {'metric':<14}{'LI (this run)':>15}{'HI (recorded)':>15}")
print("  " + "-" * 44)
hi_ref = {"auroc": 0.8991, "auroc_shuf": 0.4975, "p@100": 0.8740,
          "p@500": 0.7176, "p@1000": 0.5912, "p@npos": 0.4207}
for metric, hi_val in hi_ref.items():
    print(f"  {metric:<14}{res[metric].mean():>15.4f}{hi_val:>15.4f}")
print(f"  {'fx_tier_rate':<14}{res['fx_tier_rate'].mean():>15.4f}{0.974:>15.4f}")
print(f"  {'base rate':<14}{base:>15.4%}{'1.2342%':>15}")
print(f"  lift@100 on LI: {res['p@100'].mean()/base:.1f}x "
      f"(HI was {0.8740/0.012342:.1f}x)")
print()
print("HI-Medium, frozen recipe. Reference: HI-Small 0.8991 AUROC /")
print("87.40% p@100; true LI-Small 0.8075 AUROC / 47.00% p@100.")
