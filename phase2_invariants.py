"""Phase 2 -- zero-threshold structural invariants (MATH_IDEAS.md untried item #6,
from cyber/invariants.py's idea: rules that must ALWAYS hold, self-validated by
firing rate, immune to threshold-tuning by construction).

Audit of candidate invariants on this dataset (labels untouched during rule
definition -- see the audit section's printed firing rates):

  1. Same-currency transactions pay what they receive.  HOLDS UNIVERSALLY
     (0 violations in 5,006,175 rows) -- generator-enforced, fires never,
     cannot discriminate. Reported for honesty, not used.
  2. Reinvestment format is always a self-loop.  HOLDS UNIVERSALLY
     (0 violations in 481,056 rows) -- same story.
  3. Cross-currency transactions imply a consistent FX rate.  LIVE: 99% of
     transactions sit within ~0.1% of their currency pair's median implied
     rate, but rare gross outliers exist (e.g. 0.6667 and 1.0000 on
     USD->Euro, median 0.8534 -- 22% off market, suspiciously round).

Feature: per-transaction relative deviation |implied_rate / pair_median - 1|;
per-account max_fx_dev = max over all its cross-currency transactions (either
side), 0 if none. Continuous -- no threshold anywhere. Pair medians computed
from amounts only, never labels.

Known contaminant, stated in advance: Bitcoin's implied rate genuinely spans
2x (real volatility across the dataset's time span, which this project
deliberately does not model), so Bitcoin-pair deviations will fire innocently.

Sparse-feature honesty (FINDINGS.md): accounts with any cross-currency
transaction are a minority -- population sizes and within-population rates are
reported alongside the protocol table, and precision@k is read with the
degenerate-tie warning in mind.

Scored: max_fx_dev alone (full protocol), then marginal value on top of the
operational best (static + hub-excluded 1-hop CF), 5 seeds, operational k.

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
print("LOAD")
tx = pd.concat([pd.read_parquet(f) for f in FILES], ignore_index=True)
tx["from_acct"] = tx["From Bank"].astype(str) + ":" + tx["Account"]
tx["to_acct"] = tx["To Bank"].astype(str) + ":" + tx["Account.1"]
print(f"  combined: {len(tx):,} rows")

all_accts = sorted(set(tx["from_acct"]) | set(tx["to_acct"]))
n = len(all_accts)
idx_of = {a: i for i, a in enumerate(all_accts)}

flagged = tx[tx["Is Laundering"] == 1]
pos_accts = set(flagged["from_acct"]) | set(flagged["to_acct"])
label = pd.Index(all_accts).isin(pos_accts).astype(int)
n_pos = int(label.sum())
print(f"  accounts: {n:,}   positive: {n_pos:,} = {n_pos/n:.4%}")

print()
print("=" * 70)
print("INVARIANT AUDIT (rule definition reads amounts/currencies/formats, never labels)")
same_cur = tx["Payment Currency"] == tx["Receiving Currency"]
v1 = int((same_cur & (tx["Amount Paid"] != tx["Amount Received"])).sum())
print(f"  1. same-currency Paid != Received: {v1:,} violations / {int(same_cur.sum()):,} rows"
      f"  -> {'DEAD (never fires)' if v1 == 0 else 'live'}")
reinv = tx["Payment Format"] == "Reinvestment"
v2 = int((reinv & (tx["from_acct"] != tx["to_acct"])).sum())
print(f"  2. Reinvestment not a self-loop:   {v2:,} violations / {int(reinv.sum()):,} rows"
      f"  -> {'DEAD (never fires)' if v2 == 0 else 'live'}")

cross = tx.loc[~same_cur].copy()
print(f"  3. FX-rate consistency: {len(cross):,} cross-currency rows ({len(cross)/len(tx):.4%})")
cross["pair"] = cross["Payment Currency"] + "->" + cross["Receiving Currency"]
cross["rate"] = cross["Amount Received"] / cross["Amount Paid"]
pair_median = cross.groupby("pair")["rate"].transform("median")
cross["fx_dev"] = (cross["rate"] / pair_median - 1.0).abs()
print(f"     deviation distribution: median {cross['fx_dev'].median():.6f}, "
      f"p90 {cross['fx_dev'].quantile(0.90):.6f}, p99 {cross['fx_dev'].quantile(0.99):.6f}, "
      f"p99.9 {cross['fx_dev'].quantile(0.999):.4f}, max {cross['fx_dev'].max():.4f}")

dev_long = pd.concat([
    cross[["from_acct", "fx_dev"]].rename(columns={"from_acct": "acct"}),
    cross[["to_acct", "fx_dev"]].rename(columns={"to_acct": "acct"}),
], ignore_index=True)
max_fx_dev = dev_long.groupby("acct")["fx_dev"].max()
feat = max_fx_dev.reindex(all_accts).fillna(0.0).to_numpy()
n_nonzero = int((feat > 0).sum())
print(f"     accounts with any cross-currency activity: {n_nonzero:,} "
      f"({n_nonzero/n:.2%}) -- SPARSE, degenerate-p@k warning applies")

print()
print("=" * 70)
print("TRAIN/TEST SPLIT (60/40, stratified, seed=0) -- identical split as before")
idxs = np.arange(n)
train_idx, test_idx = train_test_split(idxs, test_size=0.4, stratify=label, random_state=0)
y_train, y_test = label[train_idx], label[test_idx]
n_pos_test = int(y_test.sum())
rng = np.random.default_rng(0)
y_shuf = y_test.copy()
rng.shuffle(y_shuf)
print(f"  test: {len(test_idx):,} rows, {n_pos_test:,} positive")

print()
print("=" * 70)
print("PROTOCOL TABLE -- max_fx_dev alone (test set)")
s = feat[test_idx]
distinct = int(pd.Series(s).nunique())
auroc = roc_auc_score(y_test, s)
auroc_shuf = roc_auc_score(y_shuf, s)


def precision_at_k(scores, labels, k):
    """Top-k by score. Ties at the cutoff are included, so sel may exceed k."""
    threshold = np.partition(scores, -k)[-k]
    selected = scores >= threshold
    return int(labels[selected].sum()), int(selected.sum())


hits, sel = precision_at_k(s, y_test, n_pos_test)
header = (f"{'feature':<16}{'AUROC':>10}{'AUROC(shuf)':>13}"
          f"{'distinct':>10}{'prec@k':>10}{'k':>8}{'hits':>7}{'sel':>7}")
print(header)
print("-" * len(header))
print(f"{'max_fx_dev':<16}{auroc:>10.4f}{auroc_shuf:>13.4f}"
      f"{distinct:>10,}{hits/sel:>10.4f}{n_pos_test:>8,}{hits:>7,}{sel:>7,}")

nz_test = s > 0
print(f"\nwithin-population honesty (test set):")
print(f"  cross-currency accounts: {int(nz_test.sum()):,}, "
      f"positive {int(y_test[nz_test].sum()):,} = {y_test[nz_test].mean():.4%} "
      f"(base rate {y_test.mean():.4%}, lift {y_test[nz_test].mean()/y_test.mean():.2f}x)")
print(f"  descriptive deviation bands within cross-currency accounts "
      f"(NOT thresholds, description only):")
for lo, hi in [(0.0, 0.001), (0.001, 0.01), (0.01, 0.1), (0.1, np.inf)]:
    band = nz_test & (s >= lo) & (s < hi)
    nb = int(band.sum())
    pb = int(y_test[band].sum())
    rate = pb / nb if nb else float("nan")
    print(f"    dev in [{lo}, {hi}): {nb:>7,} accts, {pb:>5,} pos = {rate:.4%}")

print()
print("=" * 70)
print("MARGINAL VALUE on top of operational best (static + hub-excluded 1-hop CF)")
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
X_static = acc[["degree", "in_degree", "out_degree", "amount_out", "amount_in",
                "num_transactions", "distinct_banks", "reciprocal_count", "tri3_count"]].to_numpy()

counterparty_long = pd.concat([
    tx[["from_acct", "to_acct"]].rename(columns={"from_acct": "acct", "to_acct": "counterparty"}),
    tx[["to_acct", "from_acct"]].rename(columns={"to_acct": "acct", "from_acct": "counterparty"}),
], ignore_index=True).drop_duplicates()
popularity = counterparty_long.groupby("counterparty")["acct"].nunique()
hubs = set(popularity[popularity > HUB_CUTOFF].index)
kept_cp = counterparty_long[~counterparty_long["counterparty"].isin(hubs)]
pop_kept = kept_cp.groupby("counterparty")["acct"].nunique()
idf = 1.0 / np.sqrt(1.0 + pop_kept)
rows_ = kept_cp["acct"].map(idx_of).to_numpy()
cols_ = kept_cp["counterparty"].map(idx_of).to_numpy()
M = sp.csr_matrix((idf.reindex(kept_cp["counterparty"]).to_numpy(), (rows_, cols_)), shape=(n, n))
row_norms = np.sqrt(M.multiply(M).sum(axis=1)).A.flatten()
row_norms[row_norms == 0] = 1.0
M = sp.diags(1.0 / row_norms) @ M


def fit_combined(feats, tr, te, ytr, seed):
    X = np.column_stack(feats)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[tr])
    Xte = scaler.transform(X[te])
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]


results = {"without_fx": [], "with_fx": [], "with_fxflag": []}
for seed in SEEDS:
    tr, te = train_test_split(idxs, test_size=0.4, stratify=label, random_state=seed)
    ytr, yte = label[tr], label[te]
    npt = int(yte.sum())
    rng = np.random.default_rng(seed)
    ysh = yte.copy()
    rng.shuffle(ysh)
    u1 = M[tr].T @ ytr
    h1 = np.asarray(M @ u1).flatten()
    variants = {
        "without_fx": fit_combined([X_static, h1], tr, te, ytr, seed),
        "with_fx": fit_combined([X_static, h1, feat], tr, te, ytr, seed),
        # binary "has any cross-currency tx" -- distinct test: max_fx_dev's values
        # are mostly ~1e-6, so after standardization the zero/nonzero split is
        # invisible to the model; the 13x-lift population marker deserves its own
        # feature. Natural zero/nonzero split, not a tuned threshold.
        "with_fxflag": fit_combined([X_static, h1, (feat > 0).astype(float)], tr, te, ytr, seed),
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
        print(f"  seed {seed} {name:<11} AUROC {row['auroc']:.4f} (shuf {row['auroc_shuf']:.4f})  "
              f"p@100 {row['p@100']:.4f} (sel {row['sel@100']})  p@500 {row['p@500']:.4f}  "
              f"p@1000 {row['p@1000']:.4f}  p@npos {row['p@npos']:.4f}")

print()
print("=" * 70)
print("VERDICT (mean across 5 seeds): does either FX feature add anything to the operational best?")
ra = pd.DataFrame(results["without_fx"])
rb = pd.DataFrame(results["with_fx"])
rc = pd.DataFrame(results["with_fxflag"])
header = f"  {'metric':<12}{'without_fx':>12}{'with_fx':>12}{'with_fxflag':>13}{'flag delta':>12}"
print(header)
print("  " + "-" * (len(header) - 2))
for metric in ["auroc", "auroc_shuf", "p@100", "p@500", "p@1000", "p@npos"]:
    a, b, c = ra[metric].mean(), rb[metric].mean(), rc[metric].mean()
    print(f"  {metric:<12}{a:>12.4f}{b:>12.4f}{c:>13.4f}{c - a:>+12.4f}")
