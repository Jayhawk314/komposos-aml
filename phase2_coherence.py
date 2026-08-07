"""Phase 2 -- amount-coherence around cycles.

MATH_IDEAS.md #1 (from topology/persistent_sheaves.py): does a detected
cycle conserve dollar value hop-to-hop? A real round-robin/layering scheme
passes similar amounts around the loop (minus fees) to disguise money; an
incidental structural cycle among otherwise-unrelated accounts should show
no such agreement. This uses the amounts already carried on the 2-cycle and
3-cycle edges found in phase2_cycles.py / phase2_cycles3.py -- not their
existence, which was already tested and scored weakly on its own (0.54,
0.51 AUROC).

coherence(cycle) = 1 - (max(edge amounts) - min(edge amounts)) / max(edge amounts)
  1.0 = perfectly conserved, 0.0 = maximally divergent.

Per-account feature: the BEST (max) coherence among all cycles the account
participates in. Accounts in no cycle get a sentinel below the [0,1] range
so they rank as "least suspicious," not "missing."

Two questions asked separately:
  1. As a bulk ranker (same protocol as everything else).
  2. WITHIN the already-flagged cycle population (the honest question,
     since coherence is only defined conditional on already being in a
     cycle): does higher coherence further concentrate risk beyond just
     "has a cycle at all"?

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
SENTINEL = -1.0   # below any real coherence value [0,1] -- "not in a cycle"

print("=" * 70)
print("LOAD")
tx = pd.concat([pd.read_parquet(f) for f in FILES], ignore_index=True)
tx["from_acct"] = tx["From Bank"].astype(str) + ":" + tx["Account"]
tx["to_acct"] = tx["To Bank"].astype(str) + ":" + tx["Account.1"]

all_accts = sorted(set(tx["from_acct"]) | set(tx["to_acct"]))
n = len(all_accts)
print(f"  accounts: {n:,}")

edge_amt = tx.groupby(["from_acct", "to_acct"])["Amount Paid"].sum()
edges = tx.loc[tx["from_acct"] != tx["to_acct"], ["from_acct", "to_acct"]].drop_duplicates()
print(f"  distinct directed edges: {len(edges):,}")

print()
print("=" * 70)
print("2-CYCLE COHERENCE")
reciprocal = edges.merge(edges, left_on=["from_acct", "to_acct"],
                          right_on=["to_acct", "from_acct"], suffixes=("", "_r"))
# each row: from_acct -> to_acct, and its reverse to_acct -> from_acct also exists.
# dedupe unordered pairs (u,v) and (v,u) both appear as separate rows; keep u<v only.
reciprocal = reciprocal[reciprocal["from_acct"] < reciprocal["to_acct"]]
amt_fwd = edge_amt.reindex(list(zip(reciprocal["from_acct"], reciprocal["to_acct"]))).to_numpy()
amt_rev = edge_amt.reindex(list(zip(reciprocal["to_acct"], reciprocal["from_acct"]))).to_numpy()
mx = np.maximum(amt_fwd, amt_rev)
mn = np.minimum(amt_fwd, amt_rev)
coh2 = 1 - (mx - mn) / mx
print(f"  2-cycles found: {len(reciprocal):,}")
print(f"  coherence: mean {coh2.mean():.4f}, median {np.median(coh2):.4f}")

acct_coh2 = {}
for u, v, c in zip(reciprocal["from_acct"], reciprocal["to_acct"], coh2):
    acct_coh2[u] = max(acct_coh2.get(u, SENTINEL), c)
    acct_coh2[v] = max(acct_coh2.get(v, SENTINEL), c)

print()
print("=" * 70)
print("3-CYCLE COHERENCE")
edges2 = edges.rename(columns={"from_acct": "u", "to_acct": "v"})
hop2 = edges2.merge(edges2, left_on="v", right_on="u", suffixes=("1", "2"))
hop2 = hop2.rename(columns={"u1": "u", "v2": "w"})[["u", "v1", "w"]]
hop2 = hop2[hop2["u"] != hop2["w"]]
triangles = hop2.merge(edges2, left_on=["w", "u"], right_on=["u", "v"], suffixes=("", "_close"))
triangles = triangles[["u", "v1", "w"]].drop_duplicates().rename(columns={"v1": "v"})
print(f"  3-cycles found: {len(triangles):,}")

a1 = edge_amt.reindex(list(zip(triangles["u"], triangles["v"]))).to_numpy()
a2 = edge_amt.reindex(list(zip(triangles["v"], triangles["w"]))).to_numpy()
a3 = edge_amt.reindex(list(zip(triangles["w"], triangles["u"]))).to_numpy()
stacked = np.vstack([a1, a2, a3])
mx3 = stacked.max(axis=0)
mn3 = stacked.min(axis=0)
coh3 = 1 - (mx3 - mn3) / mx3
print(f"  coherence: mean {coh3.mean():.4f}, median {np.median(coh3):.4f}")

acct_coh3 = {}
for u, v, w, c in zip(triangles["u"], triangles["v"], triangles["w"], coh3):
    for node in (u, v, w):
        acct_coh3[node] = max(acct_coh3.get(node, SENTINEL), c)

print()
print("=" * 70)
print("BUILD PER-ACCOUNT FEATURE (best coherence across any cycle)")
acc = pd.DataFrame(index=all_accts)
acc.index.name = "acct"
acc["coh2"] = pd.Series(acct_coh2).reindex(acc.index).fillna(SENTINEL)
acc["coh3"] = pd.Series(acct_coh3).reindex(acc.index).fillna(SENTINEL)
acc["best_coherence"] = acc[["coh2", "coh3"]].max(axis=1)

n_in_any_cycle = int((acc["best_coherence"] > SENTINEL).sum())
print(f"  accounts in >=1 detected cycle: {n_in_any_cycle:,}")

flagged = tx[tx["Is Laundering"] == 1]
pos_accts = set(flagged["from_acct"]) | set(flagged["to_acct"])
acc["label"] = acc.index.isin(pos_accts).astype(int)
n_pos = int(acc["label"].sum())
print(f"  positive accounts: {n_pos:,} = {n_pos/n:.4%}")

print()
print("=" * 70)
print("TRAIN/TEST SPLIT (60/40, stratified, seed=0) -- identical split as before")
y = acc["label"].to_numpy()
idxs = np.arange(n)
train_idx, test_idx = train_test_split(idxs, test_size=0.4, stratify=y, random_state=SEED)
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
print("SCORE ON HELD-OUT TEST SET (bulk ranker, whole population)")
header = (f"{'feature':<18}{'AUROC':>10}{'AUROC(shuf)':>13}"
          f"{'distinct':>10}{'prec@k':>10}{'k':>8}{'hits':>7}{'sel':>7}")
print(header)
print("-" * len(header))
scores = acc["best_coherence"].to_numpy()[test_idx]
distinct = int(pd.Series(scores).nunique())
auroc = roc_auc_score(y_test, scores)
auroc_shuf = roc_auc_score(y_test_shuffled, scores)
hits, sel = precision_at_k(scores, y_test, n_pos_test)
prec = hits / sel
print(f"{'best_coherence':<18}{auroc:>10.4f}{auroc_shuf:>13.4f}"
      f"{distinct:>10,}{prec:>10.4f}{n_pos_test:>8,}{hits:>7,}{sel:>7,}")

print()
print("=" * 70)
print("THE HONEST QUESTION: within the already-in-a-cycle population, does")
print("higher amount-coherence further concentrate risk beyond just having a cycle?")
mask = acc["best_coherence"].to_numpy()[test_idx] > SENTINEL
n_cyc = int(mask.sum())
if n_cyc > 0:
    y_cyc = y_test[mask]
    coh_cyc = scores[mask]
    print(f"  test accounts in a cycle: {n_cyc}, positive: {int(y_cyc.sum())} "
          f"({y_cyc.mean():.4%} -- local base rate)")
    HIGH = 0.9
    high_mask = coh_cyc >= HIGH
    low_mask = ~high_mask
    for name, m in [(f"coherence >= {HIGH}", high_mask), (f"coherence < {HIGH}", low_mask)]:
        n_m = int(m.sum())
        if n_m > 0:
            rate = y_cyc[m].mean()
            print(f"  {name}: n={n_m}, positive={int(y_cyc[m].sum())}, rate={rate:.4%}")

print()
print("=" * 70)
print("Prior: combined_with_cf 0.9161 AUROC / 0.3559 prec@k")
print("       cycle3_and_bank (whole pop.) 748 accts, 14.71% rate, 11.92x lift")
