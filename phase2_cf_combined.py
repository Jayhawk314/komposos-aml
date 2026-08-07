"""Phase 2 -- fold damped collaborative filtering into the combined model.

cf_damped alone (phase2_cf.py) scored 0.8832 AUROC / 0.3405 prec@k, far
above combined_all's 0.7759 / 0.0861. This tests whether CF adds anything
ON TOP of the counting features, or subsumes them entirely -- i.e. does a
model with both do meaningfully better than CF alone, or is CF already
capturing what degree/cycles/banks captured.

Same held-out test split as every prior run. Logistic regression fit on
TRAIN only; the CF feature itself is also built using TRAIN labels only
(no leakage), exactly as in phase2_cf.py.

No try/except. If something breaks, it raises.
"""
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FILES = [os.path.join(DATA, "hi_small_0.parquet"),
         os.path.join(DATA, "hi_small_1.parquet")]

SEED = 0

print("=" * 70)
print("LOAD + BUILD FEATURES")
tx = pd.concat([pd.read_parquet(f) for f in FILES], ignore_index=True)
print(f"  combined: {len(tx):,} rows")

tx["from_acct"] = tx["From Bank"].astype(str) + ":" + tx["Account"]
tx["to_acct"] = tx["To Bank"].astype(str) + ":" + tx["Account.1"]

out_amt = tx.groupby("from_acct")["Amount Paid"].sum()
in_amt = tx.groupby("to_acct")["Amount Received"].sum()
out_cnt = tx.groupby("from_acct").size()
in_cnt = tx.groupby("to_acct").size()
in_degree = tx.groupby("to_acct")["from_acct"].nunique()
out_degree = tx.groupby("from_acct")["to_acct"].nunique()

banks_long = pd.concat([
    tx[["from_acct", "From Bank"]].rename(columns={"from_acct": "acct", "From Bank": "bank"}),
    tx[["from_acct", "To Bank"]].rename(columns={"from_acct": "acct", "To Bank": "bank"}),
    tx[["to_acct", "From Bank"]].rename(columns={"to_acct": "acct", "From Bank": "bank"}),
    tx[["to_acct", "To Bank"]].rename(columns={"to_acct": "acct", "To Bank": "bank"}),
], ignore_index=True)
distinct_banks = banks_long.groupby("acct")["bank"].nunique()

edges = tx.loc[tx["from_acct"] != tx["to_acct"], ["from_acct", "to_acct"]].drop_duplicates()
reciprocal = edges.merge(edges, left_on=["from_acct", "to_acct"],
                          right_on=["to_acct", "from_acct"])
reciprocal_count = reciprocal.groupby("from_acct_x")["to_acct_x"].nunique()

edges2 = edges.rename(columns={"from_acct": "u", "to_acct": "v"})
hop2 = edges2.merge(edges2, left_on="v", right_on="u", suffixes=("1", "2"))
hop2 = hop2.rename(columns={"u1": "u", "v2": "w"})[["u", "v1", "w"]]
hop2 = hop2[hop2["u"] != hop2["w"]]
triangles = hop2.merge(edges2, left_on=["w", "u"], right_on=["u", "v"], suffixes=("", "_close"))
triangles = triangles[["u", "v1", "w"]].drop_duplicates().rename(columns={"v1": "v"})
tri_long = pd.concat([
    triangles[["u"]].rename(columns={"u": "acct"}),
    triangles[["v"]].rename(columns={"v": "acct"}),
    triangles[["w"]].rename(columns={"w": "acct"}),
], ignore_index=True)
tri3_count = tri_long.groupby("acct").size()

all_accts = sorted(set(tx["from_acct"]) | set(tx["to_acct"]))
n = len(all_accts)
idx_of = {a: i for i, a in enumerate(all_accts)}

acc = pd.DataFrame(index=all_accts)
acc.index.name = "acct"
acc["degree"] = (in_degree.reindex(acc.index).fillna(0) + out_degree.reindex(acc.index).fillna(0))
acc["in_degree"] = in_degree.reindex(acc.index).fillna(0)
acc["out_degree"] = out_degree.reindex(acc.index).fillna(0)
acc["amount_out"] = out_amt.reindex(acc.index).fillna(0.0)
acc["amount_in"] = in_amt.reindex(acc.index).fillna(0.0)
acc["num_transactions"] = (out_cnt.reindex(acc.index).fillna(0) + in_cnt.reindex(acc.index).fillna(0))
acc["distinct_banks"] = distinct_banks.reindex(acc.index).fillna(0)
acc["reciprocal_count"] = reciprocal_count.reindex(acc.index).fillna(0)
acc["tri3_count"] = tri3_count.reindex(acc.index).fillna(0)

flagged = tx[tx["Is Laundering"] == 1]
pos_accts = set(flagged["from_acct"]) | set(flagged["to_acct"])
acc["label"] = acc.index.isin(pos_accts).astype(int)
n_pos = int(acc["label"].sum())
print(f"  accounts: {n:,}   positive: {n_pos:,} = {n_pos/n:.4%}")

print()
print("=" * 70)
print("TRAIN/TEST SPLIT (60/40, stratified, seed=0) -- identical split as before")
y = acc["label"].to_numpy()
idxs = np.arange(n)
train_idx, test_idx = train_test_split(idxs, test_size=0.4, stratify=y, random_state=SEED)
y_train = y[train_idx]
y_test = y[test_idx]
n_pos_test = int(y_test.sum())
print(f"  test: {len(test_idx):,} rows, {n_pos_test:,} positive")

print()
print("=" * 70)
print("BUILD cf_damped FEATURE (train labels only, same as phase2_cf.py)")
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
M_cf = sp.diags(1.0 / row_norms) @ M_cf

v_cf = M_cf[train_idx].T @ y_train
acc["cf_damped"] = np.asarray(M_cf @ v_cf).flatten()

feature_cols = ["degree", "in_degree", "out_degree", "amount_out", "amount_in",
                 "num_transactions", "distinct_banks", "reciprocal_count",
                 "tri3_count", "cf_damped"]

X = acc[feature_cols].to_numpy()
X_train, X_test = X[train_idx], X[test_idx]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
model.fit(X_train_s, y_train)
scores_combined = model.predict_proba(X_test_s)[:, 1]

print()
print("  fitted coefficients (on standardized features):")
for name, coef in zip(feature_cols, model.coef_[0]):
    print(f"    {name:<18}{coef:>8.4f}")

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

cf_only = acc["cf_damped"].to_numpy()[test_idx]
for name, scores in [("cf_damped (alone)", cf_only), ("combined_with_cf (logreg)", scores_combined)]:
    distinct = int(pd.Series(scores).nunique())
    auroc = roc_auc_score(y_test, scores)
    auroc_shuf = roc_auc_score(y_test_shuffled, scores)
    hits, sel = precision_at_k(scores, y_test, n_pos_test)
    prec = hits / sel
    print(f"{name:<22}{auroc:>10.4f}{auroc_shuf:>13.4f}"
          f"{distinct:>10,}{prec:>10.4f}{n_pos_test:>8,}{hits:>7,}{sel:>7,}")

print()
print("=" * 70)
print("Prior: combined_all (no cf) 0.7759 AUROC / 0.0861 prec@k")
print("       cf_damped alone (phase2_cf.py) 0.8832 AUROC / 0.3405 prec@k")
