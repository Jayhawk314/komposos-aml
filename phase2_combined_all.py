"""Phase 2, combining -- fold every counting/graph feature found so far into
one model: degree, in_degree, out_degree, reciprocal_count (2-cycle),
tri3_count (3-cycle), amount_in, amount_out, in_ratio, num_transactions,
distinct_banks.

Still arithmetic, not structure-aware in the categorical sense -- a logistic
regression is a weighted sum. Point of this run: is there anything left in
the cheap features that combined logreg (0.7762 AUROC, phase1_combined.py)
didn't already capture, before moving to the bigger typology-matrix build.

Same held-out test split as every prior run (seed=0, 60/40, stratified,
same account ordering).

No try/except. If something breaks, it raises.
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FILES = [os.path.join(DATA, "hi_small_0.parquet"),
         os.path.join(DATA, "hi_small_1.parquet")]

SEED = 0

print("=" * 70)
print("LOAD + BUILD PER-ACCOUNT TABLE (all features so far)")
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

counterparty_long = pd.concat([
    tx[["from_acct", "to_acct"]].rename(columns={"from_acct": "acct", "to_acct": "counterparty"}),
    tx[["to_acct", "from_acct"]].rename(columns={"to_acct": "acct", "from_acct": "counterparty"}),
], ignore_index=True)
degree = counterparty_long.groupby("acct")["counterparty"].nunique()

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

flagged = tx[tx["Is Laundering"] == 1]
pos_accts = set(flagged["from_acct"]) | set(flagged["to_acct"])
all_accts = set(tx["from_acct"]) | set(tx["to_acct"])

acc = pd.DataFrame(index=sorted(all_accts))
acc.index.name = "acct"
acc["degree"] = degree.reindex(acc.index).fillna(0)
acc["in_degree"] = in_degree.reindex(acc.index).fillna(0)
acc["out_degree"] = out_degree.reindex(acc.index).fillna(0)
acc["amount_out"] = out_amt.reindex(acc.index).fillna(0.0)
acc["amount_in"] = in_amt.reindex(acc.index).fillna(0.0)
acc["in_ratio"] = acc["amount_in"] / (acc["amount_in"] + acc["amount_out"])
acc["num_transactions"] = (out_cnt.reindex(acc.index).fillna(0)
                            + in_cnt.reindex(acc.index).fillna(0))
acc["distinct_banks"] = distinct_banks.reindex(acc.index).fillna(0)
acc["reciprocal_count"] = reciprocal_count.reindex(acc.index).fillna(0)
acc["tri3_count"] = tri3_count.reindex(acc.index).fillna(0)
acc["label"] = acc.index.isin(pos_accts).astype(int)

n_accts = len(acc)
n_pos = int(acc["label"].sum())
print(f"  accounts: {n_accts:,}   positive: {n_pos:,} = {n_pos/n_accts:.4%}")

feature_cols = ["degree", "in_degree", "out_degree", "amount_out", "amount_in",
                 "in_ratio", "num_transactions", "distinct_banks",
                 "reciprocal_count", "tri3_count"]

print()
print("=" * 70)
print("TRAIN/TEST SPLIT (60/40, stratified, seed=0) -- identical split as before")
X = acc[feature_cols].to_numpy()
y = acc["label"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=SEED)
print(f"  train: {len(y_train):,} rows, {y_train.sum():,} positive")
print(f"  test:  {len(y_test):,} rows, {y_test.sum():,} positive")

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


n_pos_test = int(y_test.sum())
distinct_combined = int(pd.Series(scores_combined).nunique())
auroc_combined = roc_auc_score(y_test, scores_combined)
auroc_combined_shuf = roc_auc_score(y_test_shuffled, scores_combined)
hits_c, sel_c = precision_at_k(scores_combined, y_test, n_pos_test)
prec_c = hits_c / sel_c

print()
print("=" * 70)
print("SCORE ON HELD-OUT TEST SET")
header = (f"{'feature':<24}{'AUROC':>10}{'AUROC(shuf)':>13}"
          f"{'distinct':>10}{'prec@k':>10}{'k':>8}{'hits':>7}{'sel':>7}")
print(header)
print("-" * len(header))
print(f"{'combined_all (logreg)':<24}{auroc_combined:>10.4f}{auroc_combined_shuf:>13.4f}"
      f"{distinct_combined:>10,}{prec_c:>10.4f}{n_pos_test:>8,}{hits_c:>7,}{sel_c:>7,}")

print()
print("=" * 70)
print("Prior number for this to beat:")
print("  combined (6 features, no cycles)   0.7762 AUROC / 0.0653 prec@k   (phase1_combined.py)")
