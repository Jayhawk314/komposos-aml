"""Phase 2 -- isolate the round-robin finding.

phase2_typologies.py's fuzzy Jaccard match buried the strongest pocket found
this session (round_robin_layering: 1,016 accounts, 90 positive, 7.18x lift)
inside a bulk score that scored 0.5871 AUROC overall. This tests the
underlying boolean combination directly, without the Jaccard dilution, on
the same held-out test split as every prior run.

Four variants of "cycle + multi-bank":
  cycle2_and_bank    has_2cycle AND multi_bank
  cycle3_and_bank    has_3cycle AND multi_bank
  cycle_or_and_bank  (has_2cycle OR has_3cycle) AND multi_bank
  cycle_and_and_bank has_2cycle AND has_3cycle AND multi_bank  (literal signature)

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

print("=" * 70)
print("LOAD + BUILD PER-ACCOUNT STRUCTURE")
tx = pd.concat([pd.read_parquet(f) for f in FILES], ignore_index=True)
print(f"  combined: {len(tx):,} rows")

tx["from_acct"] = tx["From Bank"].astype(str) + ":" + tx["Account"]
tx["to_acct"] = tx["To Bank"].astype(str) + ":" + tx["Account.1"]

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

all_accts = set(tx["from_acct"]) | set(tx["to_acct"])
acc = pd.DataFrame(index=sorted(all_accts))
acc.index.name = "acct"
acc["distinct_banks"] = distinct_banks.reindex(acc.index).fillna(0)
acc["reciprocal_count"] = reciprocal_count.reindex(acc.index).fillna(0)
acc["tri3_count"] = tri3_count.reindex(acc.index).fillna(0)
n_accts = len(acc)

acc["has_2cycle"] = acc["reciprocal_count"] > 0
acc["has_3cycle"] = acc["tri3_count"] > 0
acc["multi_bank"] = acc["distinct_banks"] > 1

acc["cycle2_and_bank"] = (acc["has_2cycle"] & acc["multi_bank"]).astype(int)
acc["cycle3_and_bank"] = (acc["has_3cycle"] & acc["multi_bank"]).astype(int)
acc["cycle_or_and_bank"] = ((acc["has_2cycle"] | acc["has_3cycle"]) & acc["multi_bank"]).astype(int)
acc["cycle_and_and_bank"] = (acc["has_2cycle"] & acc["has_3cycle"] & acc["multi_bank"]).astype(int)

flagged = tx[tx["Is Laundering"] == 1]
pos_accts = set(flagged["from_acct"]) | set(flagged["to_acct"])
acc["label"] = acc.index.isin(pos_accts).astype(int)
n_pos = int(acc["label"].sum())
print(f"  accounts: {n_accts:,}   positive: {n_pos:,} = {n_pos/n_accts:.4%}")

for name in ["cycle2_and_bank", "cycle3_and_bank", "cycle_or_and_bank", "cycle_and_and_bank"]:
    n_flagged_feat = int(acc[name].sum())
    n_pos_in_feat = int(acc.loc[acc[name] == 1, "label"].sum())
    rate = n_pos_in_feat / n_flagged_feat if n_flagged_feat else float("nan")
    lift = rate / (n_pos / n_accts) if n_flagged_feat else float("nan")
    print(f"  {name:<20} n={n_flagged_feat:>7,}  positive={n_pos_in_feat:>5,}  "
          f"rate={rate:.4%}  lift={lift:.2f}x")

print()
print("=" * 70)
print("TRAIN/TEST SPLIT (60/40, stratified, seed=0) -- identical split as before")
y = acc["label"].to_numpy()
idx = np.arange(n_accts)
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
for name in ["cycle2_and_bank", "cycle3_and_bank", "cycle_or_and_bank", "cycle_and_and_bank"]:
    scores = acc[name].to_numpy()[test_idx]
    distinct = int(pd.Series(scores).nunique())
    auroc = roc_auc_score(y_test, scores)
    auroc_shuf = roc_auc_score(y_test_shuffled, scores)
    hits, sel = precision_at_k(scores, y_test, n_pos_test)
    prec = hits / sel
    print(f"{name:<20}{auroc:>10.4f}{auroc_shuf:>13.4f}"
          f"{distinct:>10,}{prec:>10.4f}{n_pos_test:>8,}{hits:>7,}{sel:>7,}")
    n_nonzero = int(scores.sum())
    n_hit_nonzero = int(y_test[scores == 1].sum())
    rate = n_hit_nonzero / n_nonzero if n_nonzero else float("nan")
    print(f"  {'':<20}(test set: {n_nonzero} accounts have this feature, "
          f"{n_hit_nonzero} positive = {rate:.4%})")

print()
print("=" * 70)
print("Prior best: combined_all (logreg) 0.7759 AUROC / 0.0861 prec@k")
