"""Phase 2 -- spectral method: weighted PageRank.

WHY.md section 3: the diagnosis for why categorical/spectral methods lost on
ATT&CK was that the graph was UNWEIGHTED, so a weighted composite degenerates
into counting. The AML transaction graph has weights (amounts). This is the
first direct test: does PageRank weighted by dollar amount separate from
PageRank weighted by plain transaction count (i.e. from counting)? If the two
come out the same, the weights aren't adding information here either.

Manual power-iteration PageRank over a sparse transition matrix, damping
0.85, dangling nodes (no out-edges) redistributed uniformly -- standard
formulation, no shortcuts taken.

Scored on the SAME held-out test split as every prior run.

No try/except. If something breaks, it raises.
"""
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FILES = [os.path.join(DATA, "hi_small_0.parquet"),
         os.path.join(DATA, "hi_small_1.parquet")]

SEED = 0
DAMPING = 0.85
MAX_ITER = 200
TOL = 1e-10

print("=" * 70)
print("LOAD")
tx = pd.concat([pd.read_parquet(f) for f in FILES], ignore_index=True)
print(f"  combined: {len(tx):,} rows")

tx["from_acct"] = tx["From Bank"].astype(str) + ":" + tx["Account"]
tx["to_acct"] = tx["To Bank"].astype(str) + ":" + tx["Account.1"]

all_accts = sorted(set(tx["from_acct"]) | set(tx["to_acct"]))
n = len(all_accts)
idx_of = {a: i for i, a in enumerate(all_accts)}
print(f"  accounts (nodes): {n:,}")

print()
print("=" * 70)
print("BUILD WEIGHTED EDGE LIST")
edge_agg = tx.groupby(["from_acct", "to_acct"]).agg(
    count=("Amount Paid", "size"), amount=("Amount Paid", "sum")).reset_index()
print(f"  distinct directed edges: {len(edge_agg):,}")

rows = edge_agg["from_acct"].map(idx_of).to_numpy()
cols = edge_agg["to_acct"].map(idx_of).to_numpy()
w_count = edge_agg["count"].to_numpy(dtype=np.float64)
w_amount = edge_agg["amount"].to_numpy(dtype=np.float64)


def make_transition(rows, cols, weights, n):
    """Row-stochastic sparse transition matrix, weight-proportional per row."""
    M = sp.csr_matrix((weights, (rows, cols)), shape=(n, n))
    row_sums = np.asarray(M.sum(axis=1)).flatten()
    dangling = row_sums == 0
    inv = np.zeros(n)
    nz = row_sums > 0
    inv[nz] = 1.0 / row_sums[nz]
    D = sp.diags(inv)
    return D @ M, dangling


def power_iteration(M, dangling, n, damping, max_iter, tol):
    r = np.full(n, 1.0 / n)
    Mt = M.transpose().tocsr()
    for it in range(max_iter):
        dangling_mass = r[dangling].sum()
        r_next = damping * (Mt @ r) + damping * dangling_mass / n + (1 - damping) / n
        diff = np.abs(r_next - r).sum()
        r = r_next
        if diff < tol:
            print(f"    converged after {it+1} iterations, L1 diff {diff:.2e}")
            return r
    print(f"    did NOT converge within {max_iter} iterations, last L1 diff {diff:.2e}")
    return r


print()
print("=" * 70)
print("POWER ITERATION -- count-weighted (transition prop. to transaction count)")
t0 = time.time()
M_count, dangling_count = make_transition(rows, cols, w_count, n)
pr_count = power_iteration(M_count, dangling_count, n, DAMPING, MAX_ITER, TOL)
print(f"  {time.time()-t0:.1f}s   sum={pr_count.sum():.6f}   "
      f"dangling nodes: {int(dangling_count.sum()):,}")

print()
print("POWER ITERATION -- amount-weighted (transition prop. to dollar amount)")
t0 = time.time()
M_amount, dangling_amount = make_transition(rows, cols, w_amount, n)
pr_amount = power_iteration(M_amount, dangling_amount, n, DAMPING, MAX_ITER, TOL)
print(f"  {time.time()-t0:.1f}s   sum={pr_amount.sum():.6f}   "
      f"dangling nodes: {int(dangling_amount.sum()):,}")

correlation = np.corrcoef(pr_count, pr_amount)[0, 1]
print()
print(f"  Pearson correlation, count-weighted vs amount-weighted PageRank: {correlation:.4f}")

flagged = tx[tx["Is Laundering"] == 1]
pos_accts = set(flagged["from_acct"]) | set(flagged["to_acct"])
label = np.array([1 if a in pos_accts else 0 for a in all_accts])
n_pos = int(label.sum())
print(f"  positive accounts: {n_pos:,} = {n_pos/n:.4%}")

print()
print("=" * 70)
print("TRAIN/TEST SPLIT (60/40, stratified, seed=0) -- identical split as before")
idxs = np.arange(n)
train_idx, test_idx = train_test_split(idxs, test_size=0.4, stratify=label, random_state=SEED)
y_test = label[test_idx]
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
header = (f"{'feature':<24}{'AUROC':>10}{'AUROC(shuf)':>13}"
          f"{'distinct':>10}{'prec@k':>10}{'k':>8}{'hits':>7}{'sel':>7}")
print(header)
print("-" * len(header))
for name, pr in [("pagerank_count", pr_count), ("pagerank_amount", pr_amount)]:
    scores = pr[test_idx]
    distinct = int(pd.Series(scores).nunique())
    auroc = roc_auc_score(y_test, scores)
    auroc_shuf = roc_auc_score(y_test_shuffled, scores)
    hits, sel = precision_at_k(scores, y_test, n_pos_test)
    prec = hits / sel
    print(f"{name:<24}{auroc:>10.4f}{auroc_shuf:>13.4f}"
          f"{distinct:>10,}{prec:>10.4f}{n_pos_test:>8,}{hits:>7,}{sel:>7,}")

print()
print("=" * 70)
print("Prior best: combined_all (logreg) 0.7759 AUROC / 0.0861 prec@k")
