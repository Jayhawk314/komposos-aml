"""Phase 2, step 3 -- 3-cycles.

Round-robin through one middleman: A pays B, B pays C, C pays A. Extends
phase2_cycles.py's 2-cycle (reciprocal edge) result, which scored 0.5390
AUROC / 0.0857 precision@k -- weak overall ranking, but the best precision@k
seen so far, because almost nobody (6,683 of 515,088 accounts) has one.

Computed as two merges over the deduped directed edge list:
  1. 2-hop paths: u->v->w  (edges joined to edges on v == u2)
  2. close the triangle: keep only paths where w->u also exists

Sizes are printed at each step -- a directed-path join can blow up on a
dense graph, and this one is not assumed dense in advance.

Scored on the SAME held-out test split as the earlier phase1/phase2 runs.

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
print("3-CYCLE FEATURE: A->B->C->A")
edges = tx.loc[tx["from_acct"] != tx["to_acct"], ["from_acct", "to_acct"]].drop_duplicates()
edges = edges.rename(columns={"from_acct": "u", "to_acct": "v"})
print(f"  distinct directed edges (self-loops excluded): {len(edges):,}")

hop2 = edges.merge(edges, left_on="v", right_on="u", suffixes=("1", "2"))
hop2 = hop2.rename(columns={"u1": "u", "v2": "w"})[["u", "v1", "w"]]
hop2 = hop2[hop2["u"] != hop2["w"]]  # exclude 2-cycles masquerading as u->v->u
print(f"  2-hop paths u->v->w (w != u): {len(hop2):,}")

triangles = hop2.merge(edges, left_on=["w", "u"], right_on=["u", "v"], suffixes=("", "_close"))
triangles = triangles[["u", "v1", "w"]].drop_duplicates()
triangles = triangles.rename(columns={"v1": "v"})
print(f"  closed 3-cycles u->v->w->u: {len(triangles):,}")

tri_long = pd.concat([
    triangles[["u"]].rename(columns={"u": "acct"}),
    triangles[["v"]].rename(columns={"v": "acct"}),
    triangles[["w"]].rename(columns={"w": "acct"}),
], ignore_index=True)
tri_count = tri_long.groupby("acct").size()
print(f"  accounts participating in >=1 3-cycle: {len(tri_count):,}")

acc["tri3_count"] = tri_count.reindex(acc.index).fillna(0)

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

for name in ["degree", "tri3_count"]:
    scores = acc[name].to_numpy()[test_idx]
    distinct = int(pd.Series(scores).nunique())
    auroc = roc_auc_score(y_test, scores)
    auroc_shuf = roc_auc_score(y_test_shuffled, scores)
    hits, sel = precision_at_k(scores, y_test, n_pos_test)
    prec = hits / sel
    print(f"{name:<18}{auroc:>10.4f}{auroc_shuf:>13.4f}"
          f"{distinct:>10,}{prec:>10.4f}{n_pos_test:>8,}{hits:>7,}{sel:>9,}")

print()
print("  tri3_count prec@k above is DEGENERATE: k (2,543) exceeds the number")
print("  of nonzero-scored test accounts, so the top-k threshold falls to 0")
print("  and 'selection' becomes the whole test set -- precision collapses")
print("  to exactly the test base rate. Honest question for a feature this")
print("  sparse: among accounts that HAVE a 3-cycle, what fraction are positive?")
nonzero_mask = acc["tri3_count"].to_numpy()[test_idx] > 0
n_nonzero = int(nonzero_mask.sum())
n_hit_nonzero = int(y_test[nonzero_mask].sum())
print(f"  test accounts with tri3_count > 0: {n_nonzero}")
if n_nonzero > 0:
    print(f"  of those, positive: {n_hit_nonzero} = {n_hit_nonzero/n_nonzero:.4%}")

print()
print("=" * 70)
print("Running comparison, same test split:")
print("  combined (logreg)    0.7762 AUROC / 0.0653 prec@k")
print("  degree               0.7640 AUROC / 0.0662 prec@k")
print("  in_degree            0.7391 AUROC / 0.0737 prec@k   (phase2_fan.py)")
print("  reciprocal_count     0.5390 AUROC / 0.0857 prec@k   (phase2_cycles.py)")
print("  tri3_count row above is this method, same split, same controls.")
