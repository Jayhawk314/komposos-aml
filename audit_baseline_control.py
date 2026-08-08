"""AUDIT control (2026-08-07, see AUDIT.md) -- does the operational model beat
a trivial 'rank by has_fx, then degree' baseline at operational k? And is the
FX tier's 97.8% purity explainable by base-rate composition alone?

Written by an independent auditor, not the repo's author. Protocol identical
to every other scoring script: sorted (bank:account) keys, 60/40 stratified
split per seed, tie-inclusive precision@k, shuffle control, distinct count.
Features are label-free, so per-seed evaluation just re-slices the test set.

Baselines scored:
  A. has_fx desc, then degree desc          (lexicographic)
  B. has_fx desc, then num_transactions desc
  C. degree alone (sanity anchor against the leaderboard)

Result recorded in AUDIT.md: the operational model (0.8740 p@100) beats the
best trivial baseline (0.2448 p@100) 3.6x at top-100; the top-270 test FX
accounts by degree are 19.26% positive vs 97.8% inside the model's worklist.

No try/except. If something breaks, it raises.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FILES = [os.path.join(DATA, "hi_small_0.parquet"),
         os.path.join(DATA, "hi_small_1.parquet")]
SEEDS = [0, 1, 2, 3, 4]
K_VALUES = [100, 500, 1000]

tx = pd.concat([pd.read_parquet(f) for f in FILES], ignore_index=True)
tx["from_acct"] = tx["From Bank"].astype(str) + ":" + tx["Account"]
tx["to_acct"] = tx["To Bank"].astype(str) + ":" + tx["Account.1"]
print(f"rows: {len(tx):,}")

all_accts = sorted(set(tx["from_acct"]) | set(tx["to_acct"]))
n = len(all_accts)
acct_index = pd.Index(all_accts)

flagged = tx[tx["Is Laundering"] == 1]
pos_accts = set(flagged["from_acct"]) | set(flagged["to_acct"])
label = acct_index.isin(pos_accts).astype(int)
print(f"accounts: {n:,}   positive: {int(label.sum()):,} = {label.mean():.4%}")

in_degree = tx.groupby("to_acct")["from_acct"].nunique()
out_degree = tx.groupby("from_acct")["to_acct"].nunique()
degree = (in_degree.reindex(acct_index).fillna(0)
          + out_degree.reindex(acct_index).fillna(0)).to_numpy()
out_cnt = tx.groupby("from_acct").size()
in_cnt = tx.groupby("to_acct").size()
num_tx = (out_cnt.reindex(acct_index).fillna(0)
          + in_cnt.reindex(acct_index).fillna(0)).to_numpy()

cross = tx.loc[tx["Payment Currency"] != tx["Receiving Currency"]]
fx_accts = set(cross["from_acct"]) | set(cross["to_acct"])
has_fx = acct_index.isin(fx_accts).astype(float)
print(f"fx accounts: {int(has_fx.sum()):,}")

# lexicographic scores: has_fx dominates, tiebreak by the second feature
BIG = 1e9
score_A = has_fx * BIG + degree
score_B = has_fx * BIG + num_tx
score_C = degree.astype(float)


def precision_at_k(scores, labels, k):
    """Top-k by score. Ties at the cutoff are included, so sel may exceed k."""
    threshold = np.partition(scores, -k)[-k]
    selected = scores >= threshold
    return int(labels[selected].sum()), int(selected.sum())


idxs = np.arange(n)
rows = []
for seed in SEEDS:
    tr, te = train_test_split(idxs, test_size=0.4, stratify=label, random_state=seed)
    yte = label[te]
    npt = int(yte.sum())
    rng = np.random.default_rng(seed)
    ysh = yte.copy()
    rng.shuffle(ysh)
    for name, s_all in [("fx_then_degree", score_A),
                        ("fx_then_numtx", score_B),
                        ("degree_alone", score_C)]:
        s = s_all[te]
        row = {"seed": seed, "method": name,
               "auroc": roc_auc_score(yte, s),
               "auroc_shuf": roc_auc_score(ysh, s),
               "distinct": int(pd.Series(s).nunique())}
        for k in K_VALUES + [npt]:
            key = k if k in K_VALUES else "npos"
            hits, sel = precision_at_k(s, yte, k)
            row[f"p@{key}"] = hits / sel
            row[f"sel@{key}"] = sel
        rows.append(row)
        print(f"seed {seed} {name:<16} AUROC {row['auroc']:.4f} "
              f"(shuf {row['auroc_shuf']:.4f}, distinct {row['distinct']:,})  "
              f"p@100 {row['p@100']:.4f} (sel {row['sel@100']})  "
              f"p@500 {row['p@500']:.4f} (sel {row['sel@500']})  "
              f"p@1000 {row['p@1000']:.4f} (sel {row['sel@1000']})  "
              f"p@npos {row['p@npos']:.4f}")

res = pd.DataFrame(rows)
print()
print("MEANS ACROSS 5 SEEDS")
for name, g in res.groupby("method"):
    print(f"  {name:<16} AUROC {g['auroc'].mean():.4f}  "
          f"p@100 {g['p@100'].mean():.4f}  p@500 {g['p@500'].mean():.4f}  "
          f"p@1000 {g['p@1000'].mean():.4f}  p@npos {g['p@npos'].mean():.4f}  "
          f"shuf {g['auroc_shuf'].mean():.4f}")
print()
print("repo operational model (phase2_invariants with_fxflag, 5-seed means): "
      "AUROC 0.8991  p@100 0.8740  p@500 0.7176  p@1000 0.5912")

print()
print("FX-TIER COMPOSITION CONTROL (seed 0)")
tr, te = train_test_split(idxs, test_size=0.4, stratify=label, random_state=0)
yte = label[te]
fx_te = has_fx[te] > 0
deg_te = degree[te]
print(f"  test fx accounts: {int(fx_te.sum()):,}, positive rate "
      f"{yte[fx_te].mean():.4%}")
# top-m fx accounts by degree; m=270 mirrors the repo's seed-0 fx tier size
fx_pos_idx = np.where(fx_te)[0]
order = fx_pos_idx[np.argsort(-deg_te[fx_pos_idx])]
for m in [100, 270, 500, 1000]:
    sel = order[:m]
    print(f"  top-{m} test fx accounts by degree: {int(yte[sel].sum())}/{m} "
          f"= {yte[sel].mean():.2%} positive")
print("  (repo's seed-0 fx tier inside the model's top-1000: 264/270 = 97.8%)")
