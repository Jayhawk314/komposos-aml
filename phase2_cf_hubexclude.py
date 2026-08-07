"""Phase 2 -- explicit hub exclusion in the CF matrix: the fix for the measured
top-of-ranking tie-block defect (FINDINGS.md: precision@100 4.25% vs @1000 38.88%).

Cause of the defect: accounts whose ONLY counterparty is the giant hub
70:100428660 (14,775 touchers) all score identically and pile up at the top of
the ranking as a tie block of false positives. Steeper IDF down-weighting was
already tried and failed (phase2_cf_strongidf.py): row-normalization erases any
IDF penalty for single-counterparty accounts -- exactly the failing case.
Exclusion sidesteps that: drop hub counterparty COLUMNS from the CF matrix
entirely, so a hub-only account has an empty row and scores 0.

CUTOFF -- chosen from population structure, NOT tuned on labels: the
counterparty-popularity distribution has a literal empty gap. 515,073 of
515,088 counterparties have popularity <= 93; NONE fall in 94-566; exactly 15
(all bank 70) sit at 567-14,775. Cutoff: exclude popularity > 100. Any cutoff
inside the gap selects the same 15 hubs.

SUCCESS CRITERION, decided in advance (FINDINGS.md): precision@100 improves
substantially while precision@1000 and AUROC do not degrade. Verified across
all 5 seeds, baseline and hub-excluded variants scored side by side per seed.

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
HUB_CUTOFF = 100  # exclude counterparties with popularity > this; sits in the empty 94-566 gap

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
print("BUILD CF MATRICES: baseline (all counterparties) vs hub-excluded")
counterparty_long = pd.concat([
    tx[["from_acct", "to_acct"]].rename(columns={"from_acct": "acct", "to_acct": "counterparty"}),
    tx[["to_acct", "from_acct"]].rename(columns={"to_acct": "acct", "from_acct": "counterparty"}),
], ignore_index=True).drop_duplicates()
popularity = counterparty_long.groupby("counterparty")["acct"].nunique()

hubs = popularity[popularity > HUB_CUTOFF]
print(f"  counterparties: {len(popularity):,}   excluded as hubs (popularity > {HUB_CUTOFF}): {len(hubs)}")
print(f"  max popularity among kept: {popularity[popularity <= HUB_CUTOFF].max():,}   "
      f"min among excluded: {hubs.min():,} (the empty gap between them is the cutoff's justification)")
for cp, p in hubs.sort_values(ascending=False).items():
    print(f"    excluded hub: {cp:<24}{p:>8,}")


def build_cf_matrix(cp_long):
    pop = cp_long.groupby("counterparty")["acct"].nunique()
    idf = 1.0 / np.sqrt(1.0 + pop)
    rows_ = cp_long["acct"].map(idx_of).to_numpy()
    cols_ = cp_long["counterparty"].map(idx_of).to_numpy()
    idf_vals = idf.reindex(cp_long["counterparty"]).to_numpy()
    M = sp.csr_matrix((idf_vals, (rows_, cols_)), shape=(n, n))
    norms = np.sqrt(M.multiply(M).sum(axis=1)).A.flatten()
    norms[norms == 0] = 1.0
    return sp.diags(1.0 / norms) @ M


M_base = build_cf_matrix(counterparty_long)
kept = counterparty_long[~counterparty_long["counterparty"].isin(set(hubs.index))]
M_excl = build_cf_matrix(kept)
emptied = int((np.asarray(M_excl.sum(axis=1)).flatten() == 0).sum()
              - (np.asarray(M_base.sum(axis=1)).flatten() == 0).sum())
print(f"  matrix nnz: baseline {M_base.nnz:,} -> hub-excluded {M_excl.nnz:,}")
print(f"  accounts whose row became empty (hub-only accounts, now score 0): {emptied:,}")

static_cols = ["degree", "in_degree", "out_degree", "amount_out", "amount_in",
               "num_transactions", "distinct_banks", "reciprocal_count", "tri3_count"]
X_static = acc[static_cols].to_numpy()


def precision_at_k(scores, labels, k):
    """Top-k by score. Ties at the cutoff are included, so sel may exceed k."""
    threshold = np.partition(scores, -k)[-k]
    selected = scores >= threshold
    return int(labels[selected].sum()), int(selected.sum())


idxs = np.arange(n)
results = {"baseline": [], "hub_excluded": []}

print()
print("=" * 70)
print("PER-SEED RUNS (CF rebuilt from that seed's train labels; both variants per seed)")
for seed in SEEDS:
    train_idx, test_idx = train_test_split(idxs, test_size=0.4, stratify=label,
                                            random_state=seed)
    y_train, y_test = label[train_idx], label[test_idx]
    n_pos_test = int(y_test.sum())
    rng = np.random.default_rng(seed)
    y_shuf = y_test.copy()
    rng.shuffle(y_shuf)

    for variant, M in [("baseline", M_base), ("hub_excluded", M_excl)]:
        v_cf = M[train_idx].T @ y_train
        cf_score = np.asarray(M @ v_cf).flatten()

        X = np.column_stack([X_static, cf_score])
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X[train_idx])
        X_test_s = scaler.transform(X[test_idx])

        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
        model.fit(X_train_s, y_train)
        scores = model.predict_proba(X_test_s)[:, 1]

        row = {"seed": seed,
               "auroc": roc_auc_score(y_test, scores),
               "auroc_shuf": roc_auc_score(y_shuf, scores),
               "distinct": int(pd.Series(scores).nunique()),
               "cf_auroc": roc_auc_score(y_test, cf_score[test_idx]),
               "n_pos_test": n_pos_test}
        for k in K_VALUES + [n_pos_test]:
            key = k if k in K_VALUES else "npos"
            hits, sel = precision_at_k(scores, y_test, k)
            row[f"p@{key}"] = hits / sel
            row[f"hits@{key}"] = hits
            row[f"sel@{key}"] = sel
        results[variant].append(row)
        print(f"  seed {seed} {variant:<13} AUROC {row['auroc']:.4f} (shuf {row['auroc_shuf']:.4f})  "
              f"p@100 {row['p@100']:.4f} (sel {row['sel@100']})  p@500 {row['p@500']:.4f}  "
              f"p@1000 {row['p@1000']:.4f}  p@npos {row['p@npos']:.4f}  "
              f"cf-alone AUROC {row['cf_auroc']:.4f}  distinct {row['distinct']:,}")

print()
print("=" * 70)
print("VERDICT (mean across 5 seeds) -- criterion decided in advance:")
print("p@100 improves substantially; p@1000 and AUROC do not degrade")
base_rate = n_pos / n
print()
header = f"  {'metric':<14}{'baseline':>12}{'hub_excluded':>14}{'delta':>10}"
print(header)
print("  " + "-" * (len(header) - 2))
rb = pd.DataFrame(results["baseline"])
rx = pd.DataFrame(results["hub_excluded"])
for metric in ["auroc", "auroc_shuf", "p@100", "p@500", "p@1000", "p@npos"]:
    b, x = rb[metric].mean(), rx[metric].mean()
    print(f"  {metric:<14}{b:>12.4f}{x:>14.4f}{x - b:>+10.4f}")
for metric in ["sel@100", "hits@100", "hits@500", "hits@1000"]:
    b, x = rb[metric].mean(), rx[metric].mean()
    print(f"  {metric:<14}{b:>12.1f}{x:>14.1f}{x - b:>+10.1f}")
print()
print(f"  base rate {base_rate:.4%}; lift at top 100: "
      f"baseline {rb['p@100'].mean()/base_rate:.1f}x -> hub_excluded {rx['p@100'].mean()/base_rate:.1f}x")
print()
print("Previously reported baseline (phase2_robustness.py, 5 seeds): "
      "AUROC mean 0.9130, p@100 4.25%, p@500 13.52%, p@1000 38.88%")
