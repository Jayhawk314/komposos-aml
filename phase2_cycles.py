"""Phase 2, step 1 -- cycle detection.

PLAN.md: "Round-robin laundering is literally a cycle. Count cycles each
account participates in. Cheap, and it is a graph operation."

Starting with the cheapest cycle: a 2-cycle, i.e. a reciprocal edge -- account
A pays account B AND account B pays account A (at any point, no time window --
window choice is a threshold in disguise per WHY.md).

Two features:
  reciprocal_count   number of distinct counterparties reached in both
                     directions (a true structural count)
  reciprocal_ratio   reciprocal_count / degree -- what fraction of an
                     account's counterparties are reciprocal. This is the
                     confound control: reciprocal_count alone is expected to
                     correlate with degree just because a busier account has
                     more chances at a reciprocal edge by chance. The ratio
                     asks whether cycles carry signal degree does not already
                     have.

Scored on the SAME held-out test split as phase1_combined.py (same seed,
same stratify, same account ordering -> identical split), so the comparison
is apples to apples against degree (0.7640) and combined logreg (0.7762).

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
print("LOAD + BUILD PER-ACCOUNT TABLE (same as phase1_baseline.py)")
tx = pd.concat([pd.read_parquet(f) for f in FILES], ignore_index=True)
print(f"  combined: {len(tx):,} rows")

tx["from_acct"] = tx["From Bank"].astype(str) + ":" + tx["Account"]
tx["to_acct"] = tx["To Bank"].astype(str) + ":" + tx["Account.1"]

counterparty_long = pd.concat([
    tx[["from_acct", "to_acct"]].rename(columns={"from_acct": "acct", "to_acct": "counterparty"}),
    tx[["to_acct", "from_acct"]].rename(columns={"to_acct": "acct", "from_acct": "counterparty"}),
], ignore_index=True)
degree = counterparty_long.groupby("acct")["counterparty"].nunique()

flagged = tx[tx["Is Laundering"] == 1]
pos_accts = set(flagged["from_acct"]) | set(flagged["to_acct"])
all_accts = set(tx["from_acct"]) | set(tx["to_acct"])

acc = pd.DataFrame(index=sorted(all_accts))
acc.index.name = "acct"
acc["degree"] = degree.reindex(acc.index).fillna(0)
acc["label"] = acc.index.isin(pos_accts).astype(int)

n_accts = len(acc)
n_pos = int(acc["label"].sum())
print(f"  accounts: {n_accts:,}   positive: {n_pos:,} = {n_pos/n_accts:.4%}")

print()
print("=" * 70)
print("CYCLE FEATURE: reciprocal edges (2-cycles)")
edges = tx.loc[tx["from_acct"] != tx["to_acct"], ["from_acct", "to_acct"]].drop_duplicates()
print(f"  distinct directed edges (self-loops excluded): {len(edges):,}")

reciprocal = edges.merge(edges, left_on=["from_acct", "to_acct"],
                          right_on=["to_acct", "from_acct"])
# reciprocal now holds one row per directed edge u->v whose reverse v->u
# also exists; grouping by the ORIGINAL from_acct gives, per account, the
# count of distinct counterparties reached in both directions.
reciprocal_count = reciprocal.groupby("from_acct_x")["to_acct_x"].nunique()
print(f"  accounts with >=1 reciprocal counterparty: {len(reciprocal_count):,}")

acc["reciprocal_count"] = reciprocal_count.reindex(acc.index).fillna(0)
acc["reciprocal_ratio"] = acc["reciprocal_count"] / acc["degree"].replace(0, np.nan)
acc["reciprocal_ratio"] = acc["reciprocal_ratio"].fillna(0.0)

print()
print("=" * 70)
print("TRAIN/TEST SPLIT (60/40, stratified, seed=0) -- identical to phase1_combined.py")
y = acc["label"].to_numpy()
idx = np.arange(n_accts)
train_idx, test_idx = train_test_split(idx, test_size=0.4, stratify=y,
                                        random_state=SEED)
print(f"  test: {len(test_idx):,} rows, {y[test_idx].sum():,} positive")

y_test = y[test_idx]
n_pos_test = int(y_test.sum())

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
header = (f"{'feature':<20}{'AUROC':>10}{'AUROC(shuf)':>13}"
          f"{'distinct':>10}{'prec@k':>10}{'k':>8}{'hits':>7}{'sel':>7}")
print(header)
print("-" * len(header))

for name in ["degree", "reciprocal_count", "reciprocal_ratio"]:
    scores = acc[name].to_numpy()[test_idx]
    distinct = int(pd.Series(scores).nunique())
    auroc = roc_auc_score(y_test, scores)
    auroc_shuf = roc_auc_score(y_test_shuffled, scores)
    hits, sel = precision_at_k(scores, y_test, n_pos_test)
    prec = hits / sel
    print(f"{name:<20}{auroc:>10.4f}{auroc_shuf:>13.4f}"
          f"{distinct:>10,}{prec:>10.4f}{n_pos_test:>8,}{hits:>7,}{sel:>7,}")

print()
print("=" * 70)
print("Compare against phase1_combined.py on the same test split:")
print("  degree              0.7640 AUROC  (phase1_baseline / this run, should match)")
print("  combined (logreg)   0.7762 AUROC  (phase1_combined.py)")
print("  reciprocal_* rows above are this method, same split, same controls.")
