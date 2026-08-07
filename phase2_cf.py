"""Phase 2 -- damped collaborative filtering, ported from cyber/cooccurrence_predictor.py.

That file was the one PROVEN WINNER in the whole 109-file survey of the old
repo's math modules (0.4375 MRR vs 0.3995 popularity baseline on ATT&CK,
leave-one-out). Its mechanism: predict actor-technique affinity via shared-
technique similarity to other actors, with double damping -- divide by
sqrt(the actor's own repertoire size) and sqrt(the technique's popularity)
-- specifically to stop hub actors and hub techniques from dominating the
score. That damping is the anti-confound the AML PageRank run (0.54-0.57
AUROC, lost badly) never had.

Direct transplant: actor := account, technique := counterparty. Predict an
account's risk from shared-counterparty similarity to KNOWN POSITIVE
accounts in the TRAINING set only (no leakage), via:

  M[a, c] = idf(c) = 1 / sqrt(1 + popularity(c))     if account a touches counterparty c, else 0
  M row-normalized to unit L2 norm (cosine-ready)
  v = M_train.T @ y_train                             (per-counterparty weighted positive-count)
  score(a) = M[a] . v                                  (efficient: never forms the full similarity matrix)

Scored alongside a RAW (undamped, unnormalized) version of the exact same
mechanism -- "number of positive training accounts you share a counterparty
with" -- so the damping's contribution is visible, not assumed.

Scored on the SAME held-out test split as every prior run.

No try/except. If something breaks, it raises.
"""
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
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
print("BUILD BIPARTITE ACCOUNT-COUNTERPARTY MATRIX")
counterparty_long = pd.concat([
    tx[["from_acct", "to_acct"]].rename(columns={"from_acct": "acct", "to_acct": "counterparty"}),
    tx[["to_acct", "from_acct"]].rename(columns={"to_acct": "acct", "from_acct": "counterparty"}),
], ignore_index=True).drop_duplicates()
print(f"  distinct (account, counterparty) pairs: {len(counterparty_long):,}")

popularity = counterparty_long.groupby("counterparty")["acct"].nunique()
idf = 1.0 / np.sqrt(1.0 + popularity)
print(f"  counterparty popularity: median {popularity.median():.0f}, "
      f"max {popularity.max():,} ({popularity.idxmax()})")

rows = counterparty_long["acct"].map(idx_of).to_numpy()
cols = counterparty_long["counterparty"].map(idx_of).to_numpy()
idf_vals = idf.reindex(counterparty_long["counterparty"]).to_numpy()

M_raw = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
M_cf = sp.csr_matrix((idf_vals, (rows, cols)), shape=(n, n))
row_norms = np.sqrt(M_cf.multiply(M_cf).sum(axis=1)).A.flatten()
row_norms[row_norms == 0] = 1.0
M_cf = sp.diags(1.0 / row_norms) @ M_cf
print(f"  matrix built: {n:,} x {n:,}, {M_raw.nnz:,} nonzero entries")

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
y_train = label[train_idx]
y_test = label[test_idx]
n_pos_test = int(y_test.sum())
print(f"  train: {len(train_idx):,} rows, {y_train.sum():,} positive")
print(f"  test:  {len(test_idx):,} rows, {n_pos_test:,} positive")

print()
print("=" * 70)
print("SCORE (train-only labels propagated through shared counterparties)")
y_train_full = np.zeros(n)
y_train_full[train_idx] = y_train

v_raw = M_raw[train_idx].T @ y_train      # per-counterparty: count of positive train accounts touching it
score_raw = (M_raw @ v_raw)[test_idx]

v_cf = M_cf[train_idx].T @ y_train        # per-counterparty: damped weighted positive-count
score_cf = (M_cf @ v_cf)[test_idx]

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
for name, scores in [("cf_raw (undamped)", score_raw), ("cf_damped", score_cf)]:
    distinct = int(pd.Series(scores).nunique())
    auroc = roc_auc_score(y_test, scores)
    auroc_shuf = roc_auc_score(y_test_shuffled, scores)
    hits, sel = precision_at_k(scores, y_test, n_pos_test)
    prec = hits / sel
    print(f"{name:<20}{auroc:>10.4f}{auroc_shuf:>13.4f}"
          f"{distinct:>10,}{prec:>10.4f}{n_pos_test:>8,}{hits:>7,}{sel:>7,}")

print()
print("=" * 70)
print("Prior best: combined_all (logreg) 0.7759 AUROC / 0.0861 prec@k")
print("            cycle3_and_bank (whole pop.) 748 accts, 14.71% rate, 11.92x lift")
