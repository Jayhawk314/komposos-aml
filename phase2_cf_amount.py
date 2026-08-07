"""Phase 2 -- does weighting collaborative filtering by DOLLAR AMOUNT (not
just binary touch) add anything beyond the already-winning cf_damped?

This is the most direct remaining test of WHY.md's central hypothesis for
this whole project -- the AML graph has weights, unlike the ATT&CK graph --
applied to the one method that's actually winning so far, instead of to a
new method (PageRank already tested amount-weighting in isolation and lost
badly; this asks the same question of the strongest performer instead).

M[a, c] = idf(c) * log1p(total amount a<->c)   if account a touches counterparty c
        row-normalized to unit L2 norm

Same train-label-only propagation as phase2_cf.py: v = M_train.T @ y_train,
score = M @ v. Same held-out test split as every prior run.

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
tx["from_acct"] = tx["From Bank"].astype(str) + ":" + tx["Account"]
tx["to_acct"] = tx["To Bank"].astype(str) + ":" + tx["Account.1"]

all_accts = sorted(set(tx["from_acct"]) | set(tx["to_acct"]))
n = len(all_accts)
idx_of = {a: i for i, a in enumerate(all_accts)}
print(f"  accounts: {n:,}")

flagged = tx[tx["Is Laundering"] == 1]
pos_accts = set(flagged["from_acct"]) | set(flagged["to_acct"])
label = np.array([1 if a in pos_accts else 0 for a in all_accts])

idxs = np.arange(n)
train_idx, test_idx = train_test_split(idxs, test_size=0.4, stratify=label, random_state=SEED)
y_train = label[train_idx]
y_test = label[test_idx]
n_pos_test = int(y_test.sum())
print(f"  test: {len(test_idx):,} rows, {n_pos_test:,} positive")

print()
print("=" * 70)
print("BUILD AMOUNT-WEIGHTED BIPARTITE MATRIX")
# undirected account<->counterparty amount: sum of Amount Paid in EITHER direction
long_amt = pd.concat([
    tx[["from_acct", "to_acct", "Amount Paid"]].rename(
        columns={"from_acct": "acct", "to_acct": "counterparty", "Amount Paid": "amt"}),
    tx[["to_acct", "from_acct", "Amount Paid"]].rename(
        columns={"to_acct": "acct", "from_acct": "counterparty", "Amount Paid": "amt"}),
])
pair_amt = long_amt.groupby(["acct", "counterparty"])["amt"].sum().reset_index()
print(f"  distinct (account, counterparty) pairs: {len(pair_amt):,}")

popularity = pair_amt.groupby("counterparty")["acct"].nunique()
idf = 1.0 / np.sqrt(1.0 + popularity)

rows = pair_amt["acct"].map(idx_of).to_numpy()
cols = pair_amt["counterparty"].map(idx_of).to_numpy()
idf_vals = idf.reindex(pair_amt["counterparty"]).to_numpy()
amt_vals = np.log1p(pair_amt["amt"].to_numpy())

vals_amount = idf_vals * amt_vals
M_amt = sp.csr_matrix((vals_amount, (rows, cols)), shape=(n, n))
row_norms = np.sqrt(M_amt.multiply(M_amt).sum(axis=1)).A.flatten()
row_norms[row_norms == 0] = 1.0
M_amt = sp.diags(1.0 / row_norms) @ M_amt

v_amt = M_amt[train_idx].T @ y_train
score_amt = np.asarray(M_amt @ v_amt).flatten()[test_idx]

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
header = (f"{'feature':<22}{'AUROC':>10}{'AUROC(shuf)':>13}"
          f"{'distinct':>10}{'prec@k':>10}{'k':>8}{'hits':>7}{'sel':>7}")
print(header)
print("-" * len(header))
distinct = int(pd.Series(score_amt).nunique())
auroc = roc_auc_score(y_test, score_amt)
auroc_shuf = roc_auc_score(y_test_shuffled, score_amt)
hits, sel = precision_at_k(score_amt, y_test, n_pos_test)
prec = hits / sel
print(f"{'cf_amount_weighted':<22}{auroc:>10.4f}{auroc_shuf:>13.4f}"
      f"{distinct:>10,}{prec:>10.4f}{n_pos_test:>8,}{hits:>7,}{sel:>7,}")

print()
print("=" * 70)
print("Prior: cf_damped (binary touch, IDF only) 0.8832 AUROC / 0.3405 prec@k")
print("       combined_with_cf 0.9161 AUROC / 0.3559 prec@k")
