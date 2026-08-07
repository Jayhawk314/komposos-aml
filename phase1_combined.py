"""Phase 1b -- combined counting baseline.

Still arithmetic, not structure: a logistic regression over the same six
Phase 1 features (degree, amount_in, amount_out, in_ratio, num_transactions,
distinct_banks). The point is a harder, fairer number for Phase 2 to beat
than any single feature alone.

Unlike phase1_baseline.py, this is a fitted model, so it is scored on a
held-out test split -- an in-sample AUROC on a fitted model is not honest.

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
print("LOAD + BUILD PER-ACCOUNT TABLE (same as phase1_baseline.py)")
tx = pd.concat([pd.read_parquet(f) for f in FILES], ignore_index=True)
print(f"  combined: {len(tx):,} rows")

tx["from_acct"] = tx["From Bank"].astype(str) + ":" + tx["Account"]
tx["to_acct"] = tx["To Bank"].astype(str) + ":" + tx["Account.1"]

out_amt = tx.groupby("from_acct")["Amount Paid"].sum()
in_amt = tx.groupby("to_acct")["Amount Received"].sum()
out_cnt = tx.groupby("from_acct").size()
in_cnt = tx.groupby("to_acct").size()

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

flagged = tx[tx["Is Laundering"] == 1]
pos_accts = set(flagged["from_acct"]) | set(flagged["to_acct"])
all_accts = set(tx["from_acct"]) | set(tx["to_acct"])

acc = pd.DataFrame(index=sorted(all_accts))
acc.index.name = "acct"
acc["degree"] = degree.reindex(acc.index).fillna(0)
acc["amount_out"] = out_amt.reindex(acc.index).fillna(0.0)
acc["amount_in"] = in_amt.reindex(acc.index).fillna(0.0)
acc["in_ratio"] = acc["amount_in"] / (acc["amount_in"] + acc["amount_out"])
acc["num_transactions"] = (out_cnt.reindex(acc.index).fillna(0)
                            + in_cnt.reindex(acc.index).fillna(0))
acc["distinct_banks"] = distinct_banks.reindex(acc.index).fillna(0)
acc["label"] = acc.index.isin(pos_accts).astype(int)

n_accts = len(acc)
n_pos = int(acc["label"].sum())
print(f"  accounts: {n_accts:,}   positive: {n_pos:,} = {n_pos/n_accts:.4%}")

feature_cols = ["degree", "amount_out", "amount_in", "in_ratio",
                "num_transactions", "distinct_banks"]

print()
print("=" * 70)
print("TRAIN/TEST SPLIT (60/40, stratified, seed=0)")
X = acc[feature_cols].to_numpy()
y = acc["label"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=SEED)
print(f"  train: {len(y_train):,} rows, {y_train.sum():,} positive")
print(f"  test:  {len(y_test):,} rows, {y_test.sum():,} positive")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(class_weight="balanced", max_iter=1000,
                            random_state=SEED)
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

print()
print("=" * 70)
print("SCORE ON HELD-OUT TEST SET")
header = (f"{'feature':<22}{'AUROC':>10}{'AUROC(shuf)':>13}"
          f"{'distinct':>10}{'prec@k':>10}{'k':>8}{'hits':>7}{'sel':>7}")
print(header)
print("-" * len(header))

for i, name in enumerate(feature_cols):
    scores = X_test[:, i]
    distinct = int(pd.Series(scores).nunique())
    auroc = roc_auc_score(y_test, scores)
    auroc_shuf = roc_auc_score(y_test_shuffled, scores)
    hits, sel = precision_at_k(scores, y_test, n_pos_test)
    prec = hits / sel
    print(f"{name:<22}{auroc:>10.4f}{auroc_shuf:>13.4f}"
          f"{distinct:>10,}{prec:>10.4f}{n_pos_test:>8,}{hits:>7,}{sel:>7,}")

distinct_combined = int(pd.Series(scores_combined).nunique())
auroc_combined = roc_auc_score(y_test, scores_combined)
auroc_combined_shuf = roc_auc_score(y_test_shuffled, scores_combined)
hits_c, sel_c = precision_at_k(scores_combined, y_test, n_pos_test)
prec_c = hits_c / sel_c
print(f"{'combined (logreg)':<22}{auroc_combined:>10.4f}{auroc_combined_shuf:>13.4f}"
      f"{distinct_combined:>10,}{prec_c:>10.4f}{n_pos_test:>8,}{hits_c:>7,}{sel_c:>7,}")

print()
print("=" * 70)
print("Number for Phase 2 to beat: combined (logreg) row above, on this")
print("same held-out test set.")
