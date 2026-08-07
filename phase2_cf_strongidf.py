"""Test the fix: stronger hub-damping in cf_damped.

phase2_explain.py found a concrete failure mode: all top-5 false positives
traced to the SAME counterparty (70:100428660, touched by 14,775 accounts --
the largest hub in the dataset). With that many touchers, ~180 would be
positive by pure base-rate chance, so touching it carries almost no signal,
but idf = 1/sqrt(1+popularity) wasn't damping it enough to stop it dominating.

This tests idf = 1/(1+popularity) -- linear instead of sqrt -- which
penalizes a 14,775-toucher hub roughly 121x harder than the sqrt version did,
while barely changing the penalty for the small (8-26 toucher) counterparties
that drove every TRUE positive in phase2_explain.py.

Same held-out test split, same train-label-only propagation as phase2_cf.py.

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

flagged = tx[tx["Is Laundering"] == 1]
pos_accts = set(flagged["from_acct"]) | set(flagged["to_acct"])
label = np.array([1 if a in pos_accts else 0 for a in all_accts])

idxs = np.arange(n)
train_idx, test_idx = train_test_split(idxs, test_size=0.4, stratify=label, random_state=SEED)
y_train = label[train_idx]
y_test = label[test_idx]
n_pos_test = int(y_test.sum())
print(f"  test: {len(test_idx):,} rows, {n_pos_test:,} positive")

counterparty_long = pd.concat([
    tx[["from_acct", "to_acct"]].rename(columns={"from_acct": "acct", "to_acct": "counterparty"}),
    tx[["to_acct", "from_acct"]].rename(columns={"to_acct": "acct", "from_acct": "counterparty"}),
], ignore_index=True).drop_duplicates()
popularity = counterparty_long.groupby("counterparty")["acct"].nunique()
print(f"  hub check: counterparty 70:100428660 popularity = "
      f"{int(popularity.get('70:100428660', 0)):,}")

rows_ = counterparty_long["acct"].map(idx_of).to_numpy()
cols_ = counterparty_long["counterparty"].map(idx_of).to_numpy()
pop_vals = popularity.reindex(counterparty_long["counterparty"]).to_numpy()

rng = np.random.default_rng(SEED)
y_test_shuffled = y_test.copy()
rng.shuffle(y_test_shuffled)


def precision_at_k(scores, labels, k):
    threshold = np.partition(scores, -k)[-k]
    selected = scores >= threshold
    n_selected = int(selected.sum())
    n_hit = int(labels[selected].sum())
    return n_hit, n_selected


def build_and_score(idf_vals, name):
    M = sp.csr_matrix((idf_vals, (rows_, cols_)), shape=(n, n))
    row_norms = np.sqrt(M.multiply(M).sum(axis=1)).A.flatten()
    row_norms[row_norms == 0] = 1.0
    M_normed = sp.diags(1.0 / row_norms) @ M
    v = M_normed[train_idx].T @ y_train
    s = np.asarray(M_normed @ v).flatten()
    s_test = s[test_idx]

    distinct = int(pd.Series(s_test).nunique())
    auroc = roc_auc_score(y_test, s_test)
    auroc_shuf = roc_auc_score(y_test_shuffled, s_test)
    hits, sel = precision_at_k(s_test, y_test, n_pos_test)
    prec = hits / sel
    print(f"{name:<24}{auroc:>10.4f}{auroc_shuf:>13.4f}"
          f"{distinct:>10,}{prec:>10.4f}{n_pos_test:>8,}{hits:>7,}{sel:>7,}")
    return s, s_test


print()
print("=" * 70)
print("SCORE ON HELD-OUT TEST SET")
header = (f"{'feature':<24}{'AUROC':>10}{'AUROC(shuf)':>13}"
          f"{'distinct':>10}{'prec@k':>10}{'k':>8}{'hits':>7}{'sel':>7}")
print(header)
print("-" * len(header))

idf_sqrt = 1.0 / np.sqrt(1.0 + pop_vals)
s_sqrt, s_sqrt_test = build_and_score(idf_sqrt, "cf_damped (sqrt idf, prior)")

idf_linear = 1.0 / (1.0 + pop_vals)
s_lin, s_lin_test = build_and_score(idf_linear, "cf_linear_idf (fix)")

print()
print("=" * 70)
print("DOES THE FIX ACTUALLY REMOVE THE HUB FAILURE MODE?")
hub_idx = idx_of["70:100428660"]
print(f"  hub 70:100428660 raw popularity: {int(popularity['70:100428660']):,}")
print(f"  sqrt-idf value for this hub:   {1.0/np.sqrt(1.0+popularity['70:100428660']):.6f}")
print(f"  linear-idf value for this hub: {1.0/(1.0+popularity['70:100428660']):.6f}  "
      f"({(1.0/np.sqrt(1.0+popularity['70:100428660']))/(1.0/(1.0+popularity['70:100428660'])):.1f}x smaller)")

# the 5 known false-positive accounts from phase2_explain.py that scored via this hub only
fp_accounts = ["110836:811C614E0", "311215:80FC09770", "39220:81001F6B0",
               "319722:80F965410", "32071:8011393B0"]
print()
print("  the 5 hub-only false positives found by phase2_explain.py, before vs after:")
for a in fp_accounts:
    ai = idx_of[a]
    print(f"    {a}: sqrt-idf score {s_sqrt[ai]:.4f}  ->  linear-idf score {s_lin[ai]:.4f}")

print()
print("=" * 70)
print("Prior: cf_damped 0.8832 AUROC / 0.3405 prec@k")
