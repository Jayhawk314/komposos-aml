"""Phase 2 -- generalized cycle detection via strongly connected components.

Two independent research passes over the old repo's math modules both
converged on the same idea from different files: well_ordering.py's
Kahn's-algorithm residual and topology_bridge.py's persistent-homology
Betti-1 both amount to the same underlying fact -- a node is on SOME
cycle, of ANY length, iff it belongs to a non-trivial strongly connected
component (SCC, size >= 2).

The 2-cycle and 3-cycle detectors already tried (phase2_cycles.py,
phase2_cycles3.py) only see cycles of exactly that length. Real round-robin
layering can route through more than 2 or 3 intermediaries. This tests
whether extending to cycles of any length recovers more of the signal
those fixed-length detectors structurally cannot see, without inventing
any new machinery -- SCC decomposition is O(V+E), exact, and requires
no threshold choice.

Scored on the SAME held-out test split as every prior run.

No try/except. If something breaks, it raises.
"""
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FILES = [os.path.join(DATA, "hi_small_0.parquet"),
         os.path.join(DATA, "hi_small_1.parquet")]

SEED = 0

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
print("STRONGLY CONNECTED COMPONENTS (self-loops excluded, as in prior cycle runs)")
edges = tx.loc[tx["from_acct"] != tx["to_acct"], ["from_acct", "to_acct"]].drop_duplicates()
rows = edges["from_acct"].map(idx_of).to_numpy()
cols = edges["to_acct"].map(idx_of).to_numpy()
print(f"  distinct directed edges: {len(edges):,}")

adj = sp.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
n_components, labels = connected_components(adj, directed=True, connection="strong")
print(f"  strongly connected components: {n_components:,}")

comp_sizes = np.bincount(labels)
scc_size = comp_sizes[labels]              # per-node: size of its own SCC
in_nontrivial_scc = (scc_size > 1).astype(int)

n_in_scc = int(in_nontrivial_scc.sum())
print(f"  accounts in a non-trivial SCC (size>=2, i.e. on SOME cycle): {n_in_scc:,}")
top_sizes = sorted(comp_sizes[comp_sizes > 1], reverse=True)[:10]
print(f"  10 largest non-trivial SCC sizes: {top_sizes}")
print(f"  distribution: {int((comp_sizes==2).sum())} pairs, "
      f"{int((comp_sizes==3).sum())} triples, "
      f"{int(((comp_sizes>=4)&(comp_sizes<10)).sum())} size 4-9, "
      f"{int((comp_sizes>=10).sum())} size 10+")

acc = pd.DataFrame(index=all_accts)
acc.index.name = "acct"
acc["scc_size"] = scc_size
acc["in_nontrivial_scc"] = in_nontrivial_scc
# one component (17,075 accounts) dwarfs every other non-trivial SCC (max 12
# elsewhere) -- almost certainly the network's well-connected core, not a
# laundering ring. Testing whether excluding it (size>=100 cutoff, chosen
# from the observed gap in the data just printed, not tuned against the
# label) sharpens the signal instead of diluting it.
acc["in_small_scc"] = ((acc["scc_size"] > 1) & (acc["scc_size"] < 100)).astype(int)

flagged = tx[tx["Is Laundering"] == 1]
pos_accts = set(flagged["from_acct"]) | set(flagged["to_acct"])
acc["label"] = acc.index.isin(pos_accts).astype(int)
n_pos = int(acc["label"].sum())
print()
print(f"  positive accounts: {n_pos:,} = {n_pos/n:.4%}")

print()
print("=" * 70)
print("TRAIN/TEST SPLIT (60/40, stratified, seed=0) -- identical split as before")
y = acc["label"].to_numpy()
idx = np.arange(n)
train_idx, test_idx = train_test_split(idx, test_size=0.4, stratify=y, random_state=SEED)
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
header = (f"{'feature':<20}{'AUROC':>10}{'AUROC(shuf)':>13}"
          f"{'distinct':>10}{'prec@k':>10}{'k':>8}{'hits':>7}{'sel':>7}")
print(header)
print("-" * len(header))
for name in ["in_nontrivial_scc", "scc_size", "in_small_scc"]:
    scores = acc[name].to_numpy()[test_idx]
    distinct = int(pd.Series(scores).nunique())
    auroc = roc_auc_score(y_test, scores)
    auroc_shuf = roc_auc_score(y_test_shuffled, scores)
    hits, sel = precision_at_k(scores, y_test, n_pos_test)
    prec = hits / sel
    print(f"{name:<20}{auroc:>10.4f}{auroc_shuf:>13.4f}"
          f"{distinct:>10,}{prec:>10.4f}{n_pos_test:>8,}{hits:>7,}{sel:>7,}")

print()
print("Precision among accounts actually in a non-trivial SCC (honest question")
print("for a sparse feature, same as the 3-cycle diagnostic earlier):")
for label, colname in [("all non-trivial SCCs", "in_nontrivial_scc"),
                        ("small SCCs only (<100)", "in_small_scc")]:
    mask = acc[colname].to_numpy()[test_idx] == 1
    n_nonzero = int(mask.sum())
    n_hit_nonzero = int(y_test[mask].sum())
    rate = n_hit_nonzero / n_nonzero if n_nonzero else float("nan")
    lift = rate / (n_pos_test / len(test_idx)) if n_nonzero else float("nan")
    print(f"  [{label}] test accounts: {n_nonzero}, positive: {n_hit_nonzero} "
          f"= {rate:.4%}  (lift {lift:.2f}x)")

print()
print("=" * 70)
print("Prior best: combined_all (logreg) 0.7759 AUROC / 0.0861 prec@k")
print("            cycle3_and_bank (whole pop.) 748 accts, 14.71% rate, 11.92x lift")
