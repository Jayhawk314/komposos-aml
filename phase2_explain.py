"""Explain, not just rank -- for individual flagged accounts, show WHY.

Everything before this script produced a score column: AUROC, precision@k,
a number. That tells a data scientist the ranking is good. It tells an
investigator nothing about a specific account. This decomposes cf_damped's
score for individual accounts into its actual components:

  score(a) = sum over counterparties c that a touches of
             [ idf(c) / ||a|| ] * v[c]
  v[c]     = sum over TRAINING POSITIVE accounts p that touch c of
             [ idf(c) / ||p|| ]

So a flagged account's score is literally a sum of "you share counterparty
c with these specific known-bad accounts, and c is this rare/common." That
is traceable, not a black box -- this prints the trace for real examples:
some correctly flagged (true positives) and some incorrectly flagged
(false positives), so the explanation includes failure modes honestly, not
just the wins.

No try/except. If something breaks, it raises.
"""
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FILES = [os.path.join(DATA, "hi_small_0.parquet"),
         os.path.join(DATA, "hi_small_1.parquet")]

SEED = 0
N_EXAMPLES = 5

print("=" * 70)
print("LOAD + REBUILD cf_damped (same as phase2_cf.py)")
tx = pd.concat([pd.read_parquet(f) for f in FILES], ignore_index=True)
tx["from_acct"] = tx["From Bank"].astype(str) + ":" + tx["Account"]
tx["to_acct"] = tx["To Bank"].astype(str) + ":" + tx["Account.1"]

all_accts = sorted(set(tx["from_acct"]) | set(tx["to_acct"]))
n = len(all_accts)
idx_of = {a: i for i, a in enumerate(all_accts)}
acct_of = {i: a for a, i in idx_of.items()}

flagged = tx[tx["Is Laundering"] == 1]
pos_accts = set(flagged["from_acct"]) | set(flagged["to_acct"])
label = np.array([1 if a in pos_accts else 0 for a in all_accts])

idxs = np.arange(n)
train_idx, test_idx = train_test_split(idxs, test_size=0.4, stratify=label, random_state=SEED)
y_train = label[train_idx]
y_test = label[test_idx]

counterparty_long = pd.concat([
    tx[["from_acct", "to_acct"]].rename(columns={"from_acct": "acct", "to_acct": "counterparty"}),
    tx[["to_acct", "from_acct"]].rename(columns={"to_acct": "acct", "from_acct": "counterparty"}),
], ignore_index=True).drop_duplicates()
popularity = counterparty_long.groupby("counterparty")["acct"].nunique()
idf = 1.0 / np.sqrt(1.0 + popularity)

rows_ = counterparty_long["acct"].map(idx_of).to_numpy()
cols_ = counterparty_long["counterparty"].map(idx_of).to_numpy()
idf_vals = idf.reindex(counterparty_long["counterparty"]).to_numpy()
M_cf = sp.csr_matrix((idf_vals, (rows_, cols_)), shape=(n, n))
row_norms = np.sqrt(M_cf.multiply(M_cf).sum(axis=1)).A.flatten()
row_norms[row_norms == 0] = 1.0
M_cf_normed = sp.diags(1.0 / row_norms) @ M_cf

v_cf = M_cf_normed[train_idx].T @ y_train
score = np.asarray(M_cf_normed @ v_cf).flatten()

train_pos_set = set(int(i) for i in train_idx[y_train == 1])

# counterparty -> list of (train-positive account idx, its contribution to v_cf[c])
M_cf_normed_csc = M_cf_normed.tocsc()


def explain(acct_idx, top_c=3, top_neighbors=3):
    row = M_cf_normed_csc[[acct_idx], :].tocoo()
    contributions = []
    for c_idx, weight in zip(row.col, row.data):
        contrib = weight * v_cf[c_idx]
        contributions.append((c_idx, weight, contrib))
    contributions.sort(key=lambda t: -t[2])

    print(f"    total score: {score[acct_idx]:.4f}")
    print(f"    counterparties touched: {row.nnz}")
    print(f"    top contributing counterparties:")
    for c_idx, weight, contrib in contributions[:top_c]:
        c_acct = acct_of[c_idx]
        pop = int(popularity.get(c_acct, 0))
        col = M_cf_normed_csc[:, c_idx].tocoo()
        pos_touchers = [(int(r), float(v)) for r, v in zip(col.row, col.data)
                        if int(r) in train_pos_set]
        pos_touchers.sort(key=lambda t: -t[1])
        print(f"      counterparty {c_acct}  (touched by {pop:,} accounts total, "
              f"contributes {contrib:.4f} of this score)")
        for p_idx, p_weight in pos_touchers[:top_neighbors]:
            print(f"        -- shared with KNOWN POSITIVE {acct_of[p_idx]} "
                  f"(train set, also touches this counterparty)")


print()
print("=" * 70)
print(f"EXAMPLES: TRUE POSITIVES (correctly flagged, top {N_EXAMPLES} by score among actual positives)")
test_pos_idx = test_idx[y_test == 1]
test_pos_scores = score[test_pos_idx]
order = np.argsort(-test_pos_scores)
for rank, i in enumerate(order[:N_EXAMPLES], 1):
    acct_idx = int(test_pos_idx[i])
    print(f"\n  #{rank}: {acct_of[acct_idx]}  [ACTUALLY LAUNDERING -- correctly flagged]")
    explain(acct_idx)

print()
print("=" * 70)
print(f"EXAMPLES: FALSE POSITIVES (top-scored but NOT actually laundering, top {N_EXAMPLES})")
test_neg_idx = test_idx[y_test == 0]
test_neg_scores = score[test_neg_idx]
order_neg = np.argsort(-test_neg_scores)
for rank, i in enumerate(order_neg[:N_EXAMPLES], 1):
    acct_idx = int(test_neg_idx[i])
    print(f"\n  #{rank}: {acct_of[acct_idx]}  [NOT laundering -- incorrectly flagged]")
    explain(acct_idx)

print()
print("=" * 70)
print("What this shows: every flagged account's score traces to specific shared")
print("counterparties and specific known-bad accounts -- not an opaque number.")
print("The false-positive examples show what the system's mistakes actually look")
print("like, which the score table alone never surfaces.")
