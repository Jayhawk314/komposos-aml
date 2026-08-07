"""Verification check on cf_damped's 0.9161/0.3559 result (phase2_cf_combined.py).

Open question from FINDINGS.md: does the CF score's lift come mostly from
"restrict to accounts reachable from a training-set positive" (a real but
much cruder finding -- IBM's synthetic generator builds laundering rings as
literal shared-counterparty clusters, so any account sharing a counterparty
with a labelled ring member is trivially enriched), or is CF doing real
discriminative work WITHIN that reachable population?

CF's bipartite construction means its score is structurally zero for any
account that shares NO counterparty with ANY training positive -- such an
account cannot be reached by the M_cf @ v_cf computation at all. So the
right comparison is not "CF vs whole population" (already done) but
"CF's precision vs the base rate WITHIN the reachable population" --
that's the honest bar CF has to clear to be adding value beyond mere
reachability.

Reachable := same weakly-connected component (direction ignored -- money
can flow either way and ring membership doesn't care) as >=1 training-set
positive account.

No try/except. If something breaks, it raises.
"""
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FILES = [os.path.join(DATA, "hi_small_0.parquet"),
         os.path.join(DATA, "hi_small_1.parquet")]

SEED = 0

print("=" * 70)
print("LOAD")
tx = pd.concat([pd.read_parquet(f) for f in FILES], ignore_index=True)
tx["from_acct"] = tx["From Bank"].astype(str) + ":" + tx["Account"]
tx["to_acct"] = tx["To Bank"].astype(str) + ":" + tx["Account.1"]

all_accts = sorted(set(tx["from_acct"]) | set(tx["to_acct"]))
n = len(all_accts)
idx_of = {a: i for i, a in enumerate(all_accts)}
print(f"  accounts: {n:,}")

flagged = tx[tx["Is Laundering"] == 1]
pos_accts = set(flagged["from_acct"]) | set(flagged["to_acct"])
label = np.array([1 if a in pos_accts else 0 for a in all_accts])

idxs = np.arange(n)
train_idx, test_idx = train_test_split(idxs, test_size=0.4, stratify=label, random_state=SEED)
y_train = label[train_idx]
y_test = label[test_idx]
n_pos_test = int(y_test.sum())
print(f"  test: {len(test_idx):,} rows, {n_pos_test:,} positive "
      f"(base rate {n_pos_test/len(test_idx):.4%})")

print()
print("=" * 70)
print("WEAKLY CONNECTED COMPONENTS (direction ignored)")
edges = tx.loc[tx["from_acct"] != tx["to_acct"], ["from_acct", "to_acct"]].drop_duplicates()
rows = edges["from_acct"].map(idx_of).to_numpy()
cols = edges["to_acct"].map(idx_of).to_numpy()
adj = sp.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
n_wcc, wcc_labels = connected_components(adj, directed=False)
print(f"  weakly connected components: {n_wcc:,}")

train_pos_mask = np.zeros(n, dtype=bool)
train_pos_mask[train_idx[y_train == 1]] = True
train_pos_components = set(wcc_labels[train_pos_mask])
print(f"  distinct components containing >=1 training positive: {len(train_pos_components):,}")

reachable = np.isin(wcc_labels, list(train_pos_components))
n_reachable_total = int(reachable.sum())
print(f"  accounts (whole dataset) in a component with a training positive: {n_reachable_total:,} "
      f"({n_reachable_total/n:.2%} of all accounts)")

print()
print("=" * 70)
print("THE KEY COMPARISON: base rate INSIDE the reachable population")
reachable_test = reachable[test_idx]
n_reach_test = int(reachable_test.sum())
n_reach_test_pos = int(y_test[reachable_test].sum())
reach_rate = n_reach_test_pos / n_reach_test if n_reach_test else float("nan")
reach_lift = reach_rate / (n_pos_test / len(test_idx))
print(f"  test accounts reachable from a training positive: {n_reach_test:,}")
print(f"  of those, positive: {n_reach_test_pos:,} = {reach_rate:.4%}  "
      f"(lift over whole-test base rate: {reach_lift:.2f}x)")
print()
print(f"  test accounts NOT reachable from any training positive: "
      f"{len(test_idx)-n_reach_test:,}")
n_unreach_pos = n_pos_test - n_reach_test_pos
unreach_rate = n_unreach_pos / (len(test_idx)-n_reach_test) if (len(test_idx)-n_reach_test) else float("nan")
print(f"  of those, positive: {n_unreach_pos:,} = {unreach_rate:.4%}")

print()
print("  This is the number cf_damped's precision@k (0.3405) has to beat to be")
print("  doing real work beyond 'restrict to the reachable population.'")

print()
print("=" * 70)
print("RANK CF WITHIN THE REACHABLE POPULATION ONLY")
print("(if CF only knows 'reachable vs not,' it should do no better than random")
print(" WITHIN this subset -- AUROC ~0.5 here would mean the whole 0.88 AUROC was")
print(" coming from the reachable/unreachable split alone, not from CF's ranking)")

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
v_cf = M_cf[train_idx].T @ y_train
cf_score = np.asarray(M_cf @ v_cf).flatten()
cf_test = cf_score[test_idx]

# restrict to the reachable subset only
mask = reachable_test
y_sub = y_test[mask]
scores_sub = cf_test[mask]
n_sub = len(y_sub)
n_sub_pos = int(y_sub.sum())
print(f"  reachable test accounts: {n_sub:,}, positive: {n_sub_pos:,} "
      f"({n_sub_pos/n_sub:.4%} -- this is the local base rate)")

if len(set(scores_sub.tolist())) > 1 and n_sub_pos > 0 and n_sub_pos < n_sub:
    auroc_sub = roc_auc_score(y_sub, scores_sub)
    rng = np.random.default_rng(SEED)
    y_sub_shuffled = y_sub.copy()
    rng.shuffle(y_sub_shuffled)
    auroc_sub_shuf = roc_auc_score(y_sub_shuffled, scores_sub)
    print(f"  AUROC of cf_damped WITHIN the reachable population only: {auroc_sub:.4f}")
    print(f"  same, shuffled-label control: {auroc_sub_shuf:.4f}")

    k_sub = max(1, n_sub_pos)
    threshold = np.partition(scores_sub, -k_sub)[-k_sub]
    selected = scores_sub >= threshold
    hits = int(y_sub[selected].sum())
    sel = int(selected.sum())
    print(f"  precision@k WITHIN reachable population (k={k_sub}): "
          f"{hits}/{sel} = {hits/sel:.4%}  (lift over LOCAL base rate: "
          f"{(hits/sel)/(n_sub_pos/n_sub):.2f}x)")

print()
print("=" * 70)
print("Whole-test-set numbers for reference (phase2_cf.py): "
      "cf_damped 0.8832 AUROC / 0.3405 prec@k")
