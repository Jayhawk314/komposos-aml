"""Phase 1 -- counting baseline. No graph theory, no model, no window choice.

Per-account features, all plain arithmetic:
  degree             distinct counterparties (true union across directions)
  amount_in          sum of Amount Received where account is the receiver
  amount_out         sum of Amount Paid where account is the sender
  in_ratio           amount_in / (amount_in + amount_out)    [0,1], no div-by-zero
  num_transactions   count of rows where account appears as either side
  distinct_banks     distinct bank ids seen across those rows

An account identity is (Bank, Account) -- four account strings collide across
banks in file 0 alone, so Account by itself is not a safe key.

Per-account label: 1 if the account appears on either side of any transaction
flagged Is Laundering == 1, else 0.

For each feature, scored as a ranker against the label:
  - AUROC, on the real labels
  - AUROC, on labels shuffled (fixed seed)      <- shuffle control
  - distinct score values produced
  - precision@k, k = number of positive accounts, ties at the cutoff included

Plus one random-noise column scored the same way, and the class balance.

No try/except. If something breaks, it raises.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FILES = [os.path.join(DATA, "hi_small_0.parquet"),
         os.path.join(DATA, "hi_small_1.parquet")]

SEED = 0

print("=" * 70)
print("LOAD")
frames = []
for f in FILES:
    d = pd.read_parquet(f)
    print(f"  {os.path.basename(f)}: {len(d):,} rows")
    frames.append(d)
tx = pd.concat(frames, ignore_index=True)
print(f"  combined: {len(tx):,} rows")

tx["from_acct"] = tx["From Bank"].astype(str) + ":" + tx["Account"]
tx["to_acct"] = tx["To Bank"].astype(str) + ":" + tx["Account.1"]

n_tx_pos = int(tx["Is Laundering"].sum())
print(f"  transaction-level positives: {n_tx_pos:,} of {len(tx):,} = "
      f"{n_tx_pos/len(tx):.4%}")

print()
print("=" * 70)
print("BUILD PER-ACCOUNT TABLE")

# amount out / in
out_amt = tx.groupby("from_acct")["Amount Paid"].sum()
in_amt = tx.groupby("to_acct")["Amount Received"].sum()

# transaction counts on each side
out_cnt = tx.groupby("from_acct").size()
in_cnt = tx.groupby("to_acct").size()

# distinct counterparties, true union across both directions -- a
# counterparty reached via both an outgoing and incoming edge counts once
counterparty_long = pd.concat([
    tx[["from_acct", "to_acct"]].rename(columns={"from_acct": "acct", "to_acct": "counterparty"}),
    tx[["to_acct", "from_acct"]].rename(columns={"to_acct": "acct", "from_acct": "counterparty"}),
], ignore_index=True)
degree = counterparty_long.groupby("acct")["counterparty"].nunique()

# banks seen on each side (own bank appears here too, which is fine --
# "distinct banks touched" is deliberately not counterparty-only)
banks_long = pd.concat([
    tx[["from_acct", "From Bank"]].rename(columns={"from_acct": "acct", "From Bank": "bank"}),
    tx[["from_acct", "To Bank"]].rename(columns={"from_acct": "acct", "To Bank": "bank"}),
    tx[["to_acct", "From Bank"]].rename(columns={"to_acct": "acct", "From Bank": "bank"}),
    tx[["to_acct", "To Bank"]].rename(columns={"to_acct": "acct", "To Bank": "bank"}),
], ignore_index=True)
distinct_banks = banks_long.groupby("acct")["bank"].nunique()

# label: positive if the account appears on either side of a flagged tx
flagged = tx[tx["Is Laundering"] == 1]
pos_accts = set(flagged["from_acct"]) | set(flagged["to_acct"])

all_accts = set(tx["from_acct"]) | set(tx["to_acct"])
acc = pd.DataFrame(index=sorted(all_accts))
acc.index.name = "acct"

acc["amount_out"] = out_amt.reindex(acc.index).fillna(0.0)
acc["amount_in"] = in_amt.reindex(acc.index).fillna(0.0)
acc["in_ratio"] = acc["amount_in"] / (acc["amount_in"] + acc["amount_out"])

out_cnt_r = out_cnt.reindex(acc.index).fillna(0)
in_cnt_r = in_cnt.reindex(acc.index).fillna(0)
acc["num_transactions"] = out_cnt_r + in_cnt_r

acc["degree"] = degree.reindex(acc.index).fillna(0)

acc["distinct_banks"] = distinct_banks.reindex(acc.index).fillna(0)

acc["label"] = acc.index.isin(pos_accts).astype(int)

n_accts = len(acc)
n_pos = int(acc["label"].sum())
print(f"  accounts: {n_accts:,}")
print(f"  positive accounts: {n_pos:,} = {n_pos/n_accts:.4%}")

rng = np.random.default_rng(SEED)
acc["random_noise"] = rng.random(n_accts)

labels = acc["label"].to_numpy()
shuffled_labels = labels.copy()
rng2 = np.random.default_rng(SEED)
rng2.shuffle(shuffled_labels)

feature_cols = ["degree", "amount_in", "amount_out", "in_ratio",
                "num_transactions", "distinct_banks", "random_noise"]


def precision_at_k(scores, labels, k):
    threshold = np.partition(scores, -k)[-k]
    selected = scores >= threshold
    n_selected = int(selected.sum())
    n_hit = int(labels[selected].sum())
    return n_hit, n_selected


print()
print("=" * 70)
print("SCORE TABLE")
header = (f"{'feature':<18}{'AUROC':>10}{'AUROC(shuf)':>13}"
          f"{'distinct':>10}{'prec@k':>10}{'k':>8}{'hits':>7}{'sel':>7}")
print(header)
print("-" * len(header))

for col in feature_cols:
    scores = acc[col].to_numpy()
    distinct = int(pd.Series(scores).nunique())

    auroc = roc_auc_score(labels, scores)
    auroc_shuf = roc_auc_score(shuffled_labels, scores)

    hits, sel = precision_at_k(scores, labels, n_pos)
    prec = hits / sel

    print(f"{col:<18}{auroc:>10.4f}{auroc_shuf:>13.4f}"
          f"{distinct:>10,}{prec:>10.4f}{n_pos:>8,}{hits:>7,}{sel:>7,}")

print()
print("=" * 70)
print("RANDOM BASELINE (row above: random_noise)")
print(f"  expected AUROC ~0.5, expected precision@k ~ base rate "
      f"({n_pos/n_accts:.4%})")
print()
print("Deliverable stops here. Nothing built beyond this table.")
