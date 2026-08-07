"""Phase 2 -- Dempster-Shafer combination of cf_damped and degree.

MATH_IDEAS.md #4. Unlike logistic regression (phase2_cf_combined.py, which
already blends everything into one number, 0.9161 AUROC), Dempster-Shafer
keeps disagreement between sources as its own explicit quantity (the
"conflict mass" K) instead of averaging it away. This asks a different
question than the logreg combination: not "what's the best blended score,"
but "where do the two strongest detectors actively disagree, and is that
itself informative."

Per detector d, convert its score into a belief-mass function over
{Laundering, NotLaundering} with residual mass to Uncertainty:
  m_d(L)  = reliability_d * train_quantile(score_d(x))
  m_d(NL) = reliability_d * (1 - train_quantile(score_d(x)))
  m_d(U)  = 1 - reliability_d
reliability_d is a FIXED constant from each detector's already-measured
test AUROC in FINDINGS.md (degree 0.7640, cf_damped 0.8832), scaled
2*(AUROC-0.5) -- these are prior measurements, not fit on this run.

Dempster's combination rule for two sources:
  K = m1(L)m2(NL) + m1(NL)m2(L)                        (conflict mass)
  combined(L)  = [m1(L)m2(L) + m1(L)m2(U) + m1(U)m2(L)] / (1-K)
  combined(NL) = [m1(NL)m2(NL) + m1(NL)m2(U) + m1(U)m2(NL)] / (1-K)

Scored on the same held-out test split as every prior run. Quantiles are
computed from TRAIN scores only (no test-label leakage).

No try/except. If something breaks, it raises.
"""
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FILES = [os.path.join(DATA, "hi_small_0.parquet"),
         os.path.join(DATA, "hi_small_1.parquet")]

SEED = 0
REL_DEGREE = 2 * (0.7640 - 0.5)      # 0.5280 -- from FINDINGS.md, prior measurement
REL_CF = 2 * (0.8832 - 0.5)          # 0.7664 -- from FINDINGS.md, prior measurement

print("=" * 70)
print("LOAD + BUILD FEATURES")
tx = pd.concat([pd.read_parquet(f) for f in FILES], ignore_index=True)
tx["from_acct"] = tx["From Bank"].astype(str) + ":" + tx["Account"]
tx["to_acct"] = tx["To Bank"].astype(str) + ":" + tx["Account.1"]

all_accts = sorted(set(tx["from_acct"]) | set(tx["to_acct"]))
n = len(all_accts)
idx_of = {a: i for i, a in enumerate(all_accts)}

counterparty_long = pd.concat([
    tx[["from_acct", "to_acct"]].rename(columns={"from_acct": "acct", "to_acct": "counterparty"}),
    tx[["to_acct", "from_acct"]].rename(columns={"to_acct": "acct", "from_acct": "counterparty"}),
], ignore_index=True).drop_duplicates()
degree = counterparty_long.groupby("acct")["counterparty"].nunique()

flagged = tx[tx["Is Laundering"] == 1]
pos_accts = set(flagged["from_acct"]) | set(flagged["to_acct"])
label = np.array([1 if a in pos_accts else 0 for a in all_accts])
n_pos = int(label.sum())
print(f"  accounts: {n:,}   positive: {n_pos:,} = {n_pos/n:.4%}")

idxs = np.arange(n)
train_idx, test_idx = train_test_split(idxs, test_size=0.4, stratify=label, random_state=SEED)
y_train = label[train_idx]
y_test = label[test_idx]
n_pos_test = int(y_test.sum())
print(f"  test: {len(test_idx):,} rows, {n_pos_test:,} positive")

print()
print("=" * 70)
print("BUILD cf_damped (same as phase2_cf.py)")
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
cf_score = np.asarray(M_cf @ v_cf).flatten()

degree_score = degree.reindex(all_accts).fillna(0).to_numpy()

print()
print("=" * 70)
print("DEMPSTER-SHAFER COMBINATION")


def train_quantile(scores, train_idx):
    """Rank-percentile of each score against the TRAIN distribution only."""
    train_scores = scores[train_idx]
    order = np.argsort(train_scores)
    sorted_train = train_scores[order]
    q = np.searchsorted(sorted_train, scores, side="right") / len(sorted_train)
    return np.clip(q, 0.0, 1.0)


q_degree = train_quantile(degree_score, train_idx)
q_cf = train_quantile(cf_score, train_idx)

m1_L, m1_NL = REL_DEGREE * q_degree, REL_DEGREE * (1 - q_degree)
m2_L, m2_NL = REL_CF * q_cf, REL_CF * (1 - q_cf)
m1_U, m2_U = 1 - REL_DEGREE, 1 - REL_CF

K = m1_L * m2_NL + m1_NL * m2_L
combined_L = (m1_L * m2_L + m1_L * m2_U + m1_U * m2_L) / (1 - K)
combined_NL = (m1_NL * m2_NL + m1_NL * m2_U + m1_U * m2_NL) / (1 - K)

ds_score = combined_L[test_idx]
K_test = K[test_idx]
print(f"  conflict mass K: mean {K_test.mean():.4f}, max {K_test.max():.4f}")

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
distinct = int(pd.Series(ds_score).nunique())
auroc = roc_auc_score(y_test, ds_score)
auroc_shuf = roc_auc_score(y_test_shuffled, ds_score)
hits, sel = precision_at_k(ds_score, y_test, n_pos_test)
prec = hits / sel
print(f"{'dempster_shafer':<20}{auroc:>10.4f}{auroc_shuf:>13.4f}"
      f"{distinct:>10,}{prec:>10.4f}{n_pos_test:>8,}{hits:>7,}{sel:>7,}")

print()
print("=" * 70)
print("CONFLICT DIAGNOSTIC: does high inter-detector conflict correlate with the label?")
high_conflict = K_test >= np.percentile(K_test, 90)
low_conflict = ~high_conflict
for name, m in [("top-10% conflict", high_conflict), ("bottom-90% conflict", low_conflict)]:
    n_m = int(m.sum())
    rate = y_test[m].mean()
    print(f"  {name}: n={n_m}, positive rate={rate:.4%}")

print()
print("=" * 70)
print("Prior: cf_damped alone 0.8832 AUROC / 0.3405 prec@k")
print("       combined_with_cf (logreg) 0.9161 AUROC / 0.3559 prec@k")
