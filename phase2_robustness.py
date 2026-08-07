"""Robustness + operational precision for the current best method.

Two questions the leaderboard never answered:

1. IS 0.9161 A REAL NUMBER OR A LUCKY SPLIT? Every prior script used
   random_state=0 exactly once. With only 6,357 positive accounts, a single
   split can be a favourable draw. This re-runs combined_with_cf across 5
   seeds and reports the spread. A tight band means the number is real; a
   wide swing means the headline was partly luck.

2. WHAT DOES AN INVESTIGATOR ACTUALLY GET? precision@k used k=2,543 (the
   positive count) -- statistically clean, operationally meaningless, since
   nobody reviews 2,543 accounts. This reports precision@100, @500, @1000
   as well: the real "here is your worklist" number.

The CF feature is rebuilt from scratch per seed (it depends on train labels),
so there is no leakage across seeds. Static graph features are label-
independent and are built once.

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

SEEDS = [0, 1, 2, 3, 4]
K_VALUES = [100, 500, 1000]

print("=" * 70)
print("LOAD + BUILD LABEL-INDEPENDENT FEATURES (once)")
tx = pd.concat([pd.read_parquet(f) for f in FILES], ignore_index=True)
tx["from_acct"] = tx["From Bank"].astype(str) + ":" + tx["Account"]
tx["to_acct"] = tx["To Bank"].astype(str) + ":" + tx["Account.1"]
print(f"  combined: {len(tx):,} rows")

all_accts = sorted(set(tx["from_acct"]) | set(tx["to_acct"]))
n = len(all_accts)
idx_of = {a: i for i, a in enumerate(all_accts)}

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
label = acc.index.isin(pos_accts).astype(int)
n_pos = int(label.sum())
print(f"  accounts: {n:,}   positive: {n_pos:,} = {n_pos/n:.4%}")

# CF matrix structure is label-independent; only the propagation vector uses labels.
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

static_cols = ["degree", "in_degree", "out_degree", "amount_out", "amount_in",
               "num_transactions", "distinct_banks", "reciprocal_count", "tri3_count"]
X_static = acc[static_cols].to_numpy()


def precision_at_k(scores, labels, k):
    """Top-k by score. Ties at the cutoff are included, so sel may exceed k."""
    threshold = np.partition(scores, -k)[-k]
    selected = scores >= threshold
    return int(labels[selected].sum()), int(selected.sum())


idxs = np.arange(n)
results = []

print()
print("=" * 70)
print("PER-SEED RUNS (CF rebuilt from that seed's train labels each time)")
for seed in SEEDS:
    train_idx, test_idx = train_test_split(idxs, test_size=0.4, stratify=label,
                                            random_state=seed)
    y_train, y_test = label[train_idx], label[test_idx]
    n_pos_test = int(y_test.sum())

    v_cf = M_cf[train_idx].T @ y_train
    cf_score = np.asarray(M_cf @ v_cf).flatten()

    X = np.column_stack([X_static, cf_score])
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[train_idx])
    X_test_s = scaler.transform(X[test_idx])

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    model.fit(X_train_s, y_train)
    scores = model.predict_proba(X_test_s)[:, 1]

    rng = np.random.default_rng(seed)
    y_shuf = y_test.copy()
    rng.shuffle(y_shuf)

    auroc = roc_auc_score(y_test, scores)
    auroc_shuf = roc_auc_score(y_shuf, scores)
    row = {"seed": seed, "auroc": auroc, "auroc_shuf": auroc_shuf,
           "n_pos_test": n_pos_test}
    for k in K_VALUES + [n_pos_test]:
        hits, sel = precision_at_k(scores, y_test, k)
        row[f"p@{k if k in K_VALUES else 'npos'}"] = hits / sel
        row[f"hits@{k if k in K_VALUES else 'npos'}"] = hits
        row[f"sel@{k if k in K_VALUES else 'npos'}"] = sel
    results.append(row)
    print(f"  seed {seed}: AUROC {auroc:.4f} (shuf {auroc_shuf:.4f})   "
          f"p@100 {row['p@100']:.4f}   p@500 {row['p@500']:.4f}   "
          f"p@1000 {row['p@1000']:.4f}   p@npos({n_pos_test}) {row['p@npos']:.4f}")

res = pd.DataFrame(results)

print()
print("=" * 70)
print("1. IS THE HEADLINE NUMBER STABLE ACROSS SEEDS?")
print(f"  AUROC:  mean {res['auroc'].mean():.4f}   std {res['auroc'].std():.4f}   "
      f"min {res['auroc'].min():.4f}   max {res['auroc'].max():.4f}")
print(f"  shuffle control: mean {res['auroc_shuf'].mean():.4f} "
      f"(must stay ~0.50 or the harness is lying)")
print(f"  p@npos: mean {res['p@npos'].mean():.4f}   std {res['p@npos'].std():.4f}   "
      f"min {res['p@npos'].min():.4f}   max {res['p@npos'].max():.4f}")
print(f"  seed 0 alone (the previously reported headline): "
      f"AUROC {res.loc[res.seed==0,'auroc'].iloc[0]:.4f}, "
      f"p@npos {res.loc[res.seed==0,'p@npos'].iloc[0]:.4f}")

print()
print("=" * 70)
print("2. WHAT AN INVESTIGATOR ACTUALLY GETS (mean across seeds)")
base = n_pos / n
print(f"  base rate (pick at random): {base:.4%}  -- 1 in {1/base:.0f}")
print()
print(f"  {'worklist size':<16}{'precision':>12}{'hits':>8}{'reviewed':>10}{'lift':>10}")
print("  " + "-" * 54)
for k in K_VALUES:
    p = res[f"p@{k}"].mean()
    h = res[f"hits@{k}"].mean()
    s = res[f"sel@{k}"].mean()
    print(f"  top {k:<12}{p:>11.2%}{h:>8.1f}{s:>10.1f}{p/base:>9.1f}x")
p = res["p@npos"].mean()
h = res["hits@npos"].mean()
s = res["sel@npos"].mean()
print(f"  {'top ~2543':<16}{p:>11.2%}{h:>8.1f}{s:>10.1f}{p/base:>9.1f}x")

print()
print("=" * 70)
print("Previously reported (seed 0 only): 0.9161 AUROC / 0.3559 p@npos")
