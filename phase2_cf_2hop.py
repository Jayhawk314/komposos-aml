"""Phase 2 -- 2-hop collaborative filtering (MATH_IDEAS.md untried item #5).

1-hop CF (the current leader) scores account a by cosine similarity, over
shared counterparties, to KNOWN POSITIVE training accounts:

  S = M M^T          (account-account similarity; M is the IDF-damped,
                      row-normalized account x counterparty matrix)
  score1 = S @ y     (y = train labels only, zeros elsewhere)

2-hop extends the chain one step -- similarity to accounts that are
themselves similar to known positives:

  score2 = S^2 @ y = M (M^T (M (M^T y)))

computed right-to-left as four sparse matrix-vector products -- S and S^2 are
NEVER formed, so MATH_IDEAS.md's memory-blowup caution doesn't apply.

Two deliberate choices:
- HUB-EXCLUDED matrix (phase2_cf_hubexclude.py's fix, popularity > 100
  dropped, cutoff from the empty 94-566 gap in the distribution). Without it,
  nearly everything is 2 hops from everything through the 14,775-toucher
  mega-hub and the 2-hop signal would be noise; with it the operational
  precision defect stays fixed.
- score2 CONTAINS score1 (S[a,a]=1 after row normalization, so S^2 includes
  the identity path a -> a -> positive). Reported as-is; the logreg
  combination handles the overlap, and score2's marginal value is judged by
  whether adding it to the combined model beats the combined model without it.

Scored: cf_1hop alone, cf_2hop alone, combined (static + 1hop) [the current
operational best, rebuilt here as the in-run baseline], combined (static +
1hop + 2hop). Seed-0 full protocol table first, then 5-seed robustness with
operational k (100/500/1000) for the two combined variants -- the lesson from
the tie-block defect, built in from the start rather than run afterwards.

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
HUB_CUTOFF = 100

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

print()
print("=" * 70)
print("BUILD HUB-EXCLUDED CF MATRIX (popularity > 100 dropped, per phase2_cf_hubexclude.py)")
counterparty_long = pd.concat([
    tx[["from_acct", "to_acct"]].rename(columns={"from_acct": "acct", "to_acct": "counterparty"}),
    tx[["to_acct", "from_acct"]].rename(columns={"to_acct": "acct", "from_acct": "counterparty"}),
], ignore_index=True).drop_duplicates()
popularity = counterparty_long.groupby("counterparty")["acct"].nunique()
hubs = set(popularity[popularity > HUB_CUTOFF].index)
kept = counterparty_long[~counterparty_long["counterparty"].isin(hubs)]
print(f"  excluded hubs: {len(hubs)}")

pop_kept = kept.groupby("counterparty")["acct"].nunique()
idf = 1.0 / np.sqrt(1.0 + pop_kept)
rows_ = kept["acct"].map(idx_of).to_numpy()
cols_ = kept["counterparty"].map(idx_of).to_numpy()
idf_vals = idf.reindex(kept["counterparty"]).to_numpy()
M = sp.csr_matrix((idf_vals, (rows_, cols_)), shape=(n, n))
row_norms = np.sqrt(M.multiply(M).sum(axis=1)).A.flatten()
row_norms[row_norms == 0] = 1.0
M = sp.diags(1.0 / row_norms) @ M
print(f"  matrix: {n:,} x {n:,}, {M.nnz:,} nonzero")

static_cols = ["degree", "in_degree", "out_degree", "amount_out", "amount_in",
               "num_transactions", "distinct_banks", "reciprocal_count", "tri3_count"]
X_static = acc[static_cols].to_numpy()


def cf_scores(train_idx, y_train):
    """1-hop and 2-hop CF scores for ALL accounts, train labels only.
    score1 = S y, score2 = S^2 y, S = M M^T -- computed as matvecs, S never formed."""
    u1 = M[train_idx].T @ y_train        # counterparty vector
    h1 = np.asarray(M @ u1).flatten()    # 1-hop account scores
    u2 = M.T @ h1
    h2 = np.asarray(M @ u2).flatten()    # 2-hop account scores
    return h1, h2


def precision_at_k(scores, labels, k):
    """Top-k by score. Ties at the cutoff are included, so sel may exceed k."""
    threshold = np.partition(scores, -k)[-k]
    selected = scores >= threshold
    return int(labels[selected].sum()), int(selected.sum())


def fit_combined(feats, train_idx, test_idx, y_train, seed):
    X = np.column_stack(feats)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[train_idx])
    X_test_s = scaler.transform(X[test_idx])
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    model.fit(X_train_s, y_train)
    return model.predict_proba(X_test_s)[:, 1]


idxs = np.arange(n)

print()
print("=" * 70)
print("SEED 0 -- FULL PROTOCOL TABLE (same split as every other script)")
train_idx, test_idx = train_test_split(idxs, test_size=0.4, stratify=label, random_state=0)
y_train, y_test = label[train_idx], label[test_idx]
n_pos_test = int(y_test.sum())
rng = np.random.default_rng(0)
y_shuf = y_test.copy()
rng.shuffle(y_shuf)

h1, h2 = cf_scores(train_idx, y_train)
combined_1hop = fit_combined([X_static, h1], train_idx, test_idx, y_train, 0)
combined_2hop = fit_combined([X_static, h1, h2], train_idx, test_idx, y_train, 0)

rows_out = [
    ("cf_1hop alone", h1[test_idx]),
    ("cf_2hop alone", h2[test_idx]),
    ("combined_1hop", combined_1hop),
    ("combined_1hop_2hop", combined_2hop),
]
header = (f"{'feature':<22}{'AUROC':>10}{'AUROC(shuf)':>13}"
          f"{'distinct':>10}{'prec@k':>10}{'k':>8}{'hits':>7}{'sel':>7}")
print(header)
print("-" * len(header))
for name, s in rows_out:
    distinct = int(pd.Series(s).nunique())
    auroc = roc_auc_score(y_test, s)
    auroc_shuf = roc_auc_score(y_shuf, s)
    hits, sel = precision_at_k(s, y_test, n_pos_test)
    print(f"{name:<22}{auroc:>10.4f}{auroc_shuf:>13.4f}"
          f"{distinct:>10,}{hits/sel:>10.4f}{n_pos_test:>8,}{hits:>7,}{sel:>7,}")

print()
print("=" * 70)
print("5-SEED ROBUSTNESS + OPERATIONAL K (both combined variants per seed)")
results = {"combined_1hop": [], "combined_1hop_2hop": []}
for seed in SEEDS:
    tr, te = train_test_split(idxs, test_size=0.4, stratify=label, random_state=seed)
    ytr, yte = label[tr], label[te]
    npt = int(yte.sum())
    rng = np.random.default_rng(seed)
    ysh = yte.copy()
    rng.shuffle(ysh)

    s1, s2 = cf_scores(tr, ytr)
    variants = {
        "combined_1hop": fit_combined([X_static, s1], tr, te, ytr, seed),
        "combined_1hop_2hop": fit_combined([X_static, s1, s2], tr, te, ytr, seed),
    }
    for name, sc in variants.items():
        row = {"seed": seed, "auroc": roc_auc_score(yte, sc),
               "auroc_shuf": roc_auc_score(ysh, sc)}
        for k in K_VALUES + [npt]:
            key = k if k in K_VALUES else "npos"
            hits, sel = precision_at_k(sc, yte, k)
            row[f"p@{key}"] = hits / sel
            row[f"sel@{key}"] = sel
        results[name].append(row)
        print(f"  seed {seed} {name:<19} AUROC {row['auroc']:.4f} (shuf {row['auroc_shuf']:.4f})  "
              f"p@100 {row['p@100']:.4f} (sel {row['sel@100']})  p@500 {row['p@500']:.4f}  "
              f"p@1000 {row['p@1000']:.4f}  p@npos {row['p@npos']:.4f}")

print()
print("=" * 70)
print("VERDICT (mean across 5 seeds): does adding 2-hop beat 1-hop-only?")
ra = pd.DataFrame(results["combined_1hop"])
rb = pd.DataFrame(results["combined_1hop_2hop"])
header = f"  {'metric':<12}{'1hop only':>12}{'1hop+2hop':>12}{'delta':>10}"
print(header)
print("  " + "-" * (len(header) - 2))
for metric in ["auroc", "auroc_shuf", "p@100", "p@500", "p@1000", "p@npos"]:
    a, b = ra[metric].mean(), rb[metric].mean()
    print(f"  {metric:<12}{a:>12.4f}{b:>12.4f}{b - a:>+10.4f}")
print()
print("Reference (phase2_cf_hubexclude.py, 5 seeds): combined_1hop AUROC 0.8987, "
      "p@100 0.8660, p@500 0.6940, p@1000 0.5724, p@npos 0.4112")
