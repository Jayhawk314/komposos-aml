"""Phase 2 -- Kan extension, done honestly (no costume).

Over a preorder (accounts ordered by "pays"/"is paid by"), a Kan extension
is not approximated by min/max -- it IS min/max. That is the actual
mathematical fact for this special case (a preorder is a category with at
most one morphism between any two objects; limits over it are infima,
colimits are suprema). No arithmetic is being relabelled as category
theory here; this is the honest reduction.

Base signal F, defined on every account already:
  base_score(x) = 1 if x has a 2-cycle or 3-cycle (the strongest signal
  found in this session: round-robin membership), else 0.

Right Kan extension along "pays" (x -> y means x pays y):
  Ran(x) = lim_{y : x->y} F(y) = min over everyone x pays.
  Vacuous limit (x pays no one) = the TOP element = 1, by convention
  (an empty set of constraints constrains nothing).

Left Kan extension along "is paid by" (y -> x means y pays x):
  Lan(x) = colim_{y : y->x} F(y) = max over everyone who pays x.
  Vacuous colimit (nobody pays x) = the BOTTOM element = 0.

Both are one-hop propagations of the round-robin signal in each direction,
with the vacuous cases handled by the actual category-theoretic convention,
not an arbitrary fill value.

Scored on the SAME held-out test split as every prior run.

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
print("LOAD + BUILD BASE SIGNAL (round-robin: has_2cycle or has_3cycle)")
tx = pd.concat([pd.read_parquet(f) for f in FILES], ignore_index=True)
print(f"  combined: {len(tx):,} rows")

tx["from_acct"] = tx["From Bank"].astype(str) + ":" + tx["Account"]
tx["to_acct"] = tx["To Bank"].astype(str) + ":" + tx["Account.1"]

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
acc = pd.DataFrame(index=all_accts)
acc.index.name = "acct"
acc["reciprocal_count"] = reciprocal_count.reindex(acc.index).fillna(0)
acc["tri3_count"] = tri3_count.reindex(acc.index).fillna(0)
acc["base_score"] = ((acc["reciprocal_count"] > 0) | (acc["tri3_count"] > 0)).astype(int)
n_accts = len(acc)
n_base = int(acc["base_score"].sum())
print(f"  accounts: {n_accts:,}   base_score=1 (round-robin members): {n_base:,}")

print()
print("=" * 70)
print("RIGHT KAN EXTENSION (min over who x pays; vacuous -> 1)")
right = edges.merge(acc[["base_score"]], left_on="to_acct", right_index=True)
right_kan = right.groupby("from_acct")["base_score"].min()
acc["right_kan"] = right_kan.reindex(acc.index).fillna(1).astype(int)
n_has_out = int(edges["from_acct"].nunique())
print(f"  accounts with >=1 outgoing edge: {n_has_out:,} (rest get the vacuous limit, 1)")

print()
print("LEFT KAN EXTENSION (max over who pays x; vacuous -> 0)")
left = edges.merge(acc[["base_score"]], left_on="from_acct", right_index=True)
left_kan = left.groupby("to_acct")["base_score"].max()
acc["left_kan"] = left_kan.reindex(acc.index).fillna(0).astype(int)
n_has_in = int(edges["to_acct"].nunique())
print(f"  accounts with >=1 incoming edge: {n_has_in:,} (rest get the vacuous colimit, 0)")

acc["kan_or"] = ((acc["right_kan"] == 1) | (acc["left_kan"] == 1)).astype(int)

# how many NEW accounts (base_score==0) does each extension surface
new_right = int(((acc["right_kan"] == 1) & (acc["base_score"] == 0)).sum())
new_left = int(((acc["left_kan"] == 1) & (acc["base_score"] == 0)).sum())
new_or = int(((acc["kan_or"] == 1) & (acc["base_score"] == 0)).sum())
print()
print(f"  new accounts surfaced beyond base_score (not round-robin themselves):")
print(f"    right_kan: {new_right:,}   left_kan: {new_left:,}   kan_or: {new_or:,}")

flagged = tx[tx["Is Laundering"] == 1]
pos_accts = set(flagged["from_acct"]) | set(flagged["to_acct"])
acc["label"] = acc.index.isin(pos_accts).astype(int)
n_pos = int(acc["label"].sum())
print()
print(f"  positive accounts: {n_pos:,} = {n_pos/n_accts:.4%}")

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
header = (f"{'feature':<14}{'AUROC':>10}{'AUROC(shuf)':>13}"
          f"{'distinct':>10}{'prec@k':>10}{'k':>8}{'hits':>7}{'sel':>7}")
print(header)
print("-" * len(header))
for name in ["base_score", "right_kan", "left_kan", "kan_or"]:
    scores = acc[name].to_numpy()[test_idx]
    distinct = int(pd.Series(scores).nunique())
    auroc = roc_auc_score(y_test, scores)
    auroc_shuf = roc_auc_score(y_test_shuffled, scores)
    hits, sel = precision_at_k(scores, y_test, n_pos_test)
    prec = hits / sel
    print(f"{name:<14}{auroc:>10.4f}{auroc_shuf:>13.4f}"
          f"{distinct:>10,}{prec:>10.4f}{n_pos_test:>8,}{hits:>7,}{sel:>7,}")

print()
print("=" * 70)
print("Precision among newly-surfaced accounts only (base_score==0, extension==1):")
for name, mask in [
    ("right_kan (new)", (acc["right_kan"] == 1) & (acc["base_score"] == 0)),
    ("left_kan (new)", (acc["left_kan"] == 1) & (acc["base_score"] == 0)),
]:
    n_new = int(mask.sum())
    n_new_pos = int(acc.loc[mask, "label"].sum())
    rate = n_new_pos / n_new if n_new else float("nan")
    lift = rate / (n_pos / n_accts) if n_new else float("nan")
    print(f"  {name:<20} n={n_new:>7,}  positive={n_new_pos:>5,}  rate={rate:.4%}  lift={lift:.2f}x")

print()
print("=" * 70)
print("Prior best: combined_all (logreg) 0.7759 AUROC / 0.0861 prec@k")
print("            cycle3_and_bank (whole pop.) 748 accts, 14.71% rate, 11.92x lift")
