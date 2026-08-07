"""Phase 2, step 2 -- fan-in / fan-out.

Still counting, not graph theory: split the single `degree` feature into
its two directions.

  in_degree    distinct accounts that PAID this account (distinct senders)
  out_degree   distinct accounts this account PAID (distinct receivers)

Typology relevance (per PLAN.md's catalogue): a mule/fan-out pattern is
lopsided toward out_degree, a collector/fan-in pattern lopsided toward
in_degree, a pass-through/layering pattern has both high. Plain `degree`
(0.7640 AUROC, phase1/phase2 runs) cannot see that asymmetry because it
sums the two directions together.

Scored on the SAME held-out test split as phase1_combined.py / phase2_cycles.py.

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

SEED = 0

print("=" * 70)
print("LOAD + BUILD PER-ACCOUNT TABLE")
tx = pd.concat([pd.read_parquet(f) for f in FILES], ignore_index=True)
print(f"  combined: {len(tx):,} rows")

tx["from_acct"] = tx["From Bank"].astype(str) + ":" + tx["Account"]
tx["to_acct"] = tx["To Bank"].astype(str) + ":" + tx["Account.1"]

counterparty_long = pd.concat([
    tx[["from_acct", "to_acct"]].rename(columns={"from_acct": "acct", "to_acct": "counterparty"}),
    tx[["to_acct", "from_acct"]].rename(columns={"to_acct": "acct", "from_acct": "counterparty"}),
], ignore_index=True)
degree = counterparty_long.groupby("acct")["counterparty"].nunique()

# in_degree: distinct senders to this account. out_degree: distinct
# receivers from this account. Self-loops (from_acct == to_acct) count in
# both, same as they would in `degree`.
in_degree = tx.groupby("to_acct")["from_acct"].nunique()
out_degree = tx.groupby("from_acct")["to_acct"].nunique()

flagged = tx[tx["Is Laundering"] == 1]
pos_accts = set(flagged["from_acct"]) | set(flagged["to_acct"])
all_accts = set(tx["from_acct"]) | set(tx["to_acct"])

acc = pd.DataFrame(index=sorted(all_accts))
acc.index.name = "acct"
acc["degree"] = degree.reindex(acc.index).fillna(0)
acc["in_degree"] = in_degree.reindex(acc.index).fillna(0)
acc["out_degree"] = out_degree.reindex(acc.index).fillna(0)
acc["fan_out_ratio"] = acc["out_degree"] / (acc["in_degree"] + acc["out_degree"]).replace(0, np.nan)
acc["fan_out_ratio"] = acc["fan_out_ratio"].fillna(0.0)
acc["label"] = acc.index.isin(pos_accts).astype(int)

n_accts = len(acc)
n_pos = int(acc["label"].sum())
print(f"  accounts: {n_accts:,}   positive: {n_pos:,} = {n_pos/n_accts:.4%}")

print()
print("=" * 70)
print("TRAIN/TEST SPLIT (60/40, stratified, seed=0) -- identical split as before")
y = acc["label"].to_numpy()
idx = np.arange(n_accts)
train_idx, test_idx = train_test_split(idx, test_size=0.4, stratify=y,
                                        random_state=SEED)
y_test = y[test_idx]
n_pos_test = int(y_test.sum())
print(f"  test: {len(test_idx):,} rows, {n_pos_test:,} positive")

rng = np.random.default_rng(SEED)
y_test_shuffled = y_test.copy()
rng.shuffle(y_test_shuffled)


def precision_at_k(scores, labels, k):
    threshold = np.partition(scores, -k)[-k]
    selected = scores >= threshold
    n_selected = int(selected.sum())
    n_hit = int(labels[selected].sum())
    return n_hit, n_selected


print()
print("=" * 70)
print("SCORE ON HELD-OUT TEST SET")
header = (f"{'feature':<18}{'AUROC':>10}{'AUROC(shuf)':>13}"
          f"{'distinct':>10}{'prec@k':>10}{'k':>8}{'hits':>7}{'sel':>7}")
print(header)
print("-" * len(header))

for name in ["degree", "in_degree", "out_degree", "fan_out_ratio"]:
    scores = acc[name].to_numpy()[test_idx]
    distinct = int(pd.Series(scores).nunique())
    auroc = roc_auc_score(y_test, scores)
    auroc_shuf = roc_auc_score(y_test_shuffled, scores)
    hits, sel = precision_at_k(scores, y_test, n_pos_test)
    prec = hits / sel
    print(f"{name:<18}{auroc:>10.4f}{auroc_shuf:>13.4f}"
          f"{distinct:>10,}{prec:>10.4f}{n_pos_test:>8,}{hits:>7,}{sel:>7,}")

print()
print("=" * 70)
print("Running comparison, same test split:")
print("  degree               0.7640 AUROC")
print("  combined (logreg)    0.7762 AUROC   (phase1_combined.py)")
print("  reciprocal_count     0.5390 AUROC / 0.0857 prec@k   (phase2_cycles.py)")
print("  in/out/fan rows above are this method, same split, same controls.")
