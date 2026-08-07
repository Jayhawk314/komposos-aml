"""Phase 3 -- audit against IBM's typology ground truth (LI-Small_Patterns.txt).

The Kaggle archive ships, per dataset, a Patterns.txt naming every generated
laundering ATTEMPT: its typology (FAN-OUT, CYCLE, GATHER-SCATTER, ...) and its
exact transactions. Until now every typology statement in this repo was a
guess scored only against the binary label. This audits, with ground truth:

  1. Parse: 370 attempts -> per-attempt typology + involved accounts
     (bank fields are zero-padded in Patterns.txt, e.g. 021174 -- normalized
     via int() to match this repo's account key "{bank}:{account}").
  2. Coverage: are pattern accounts a subset of label-positive accounts, and
     how many positives sit in NO pattern (background laundering)?
  3. Per-typology recall of the operational-best worklist (seed 0): of each
     typology's test-set accounts, how many made the top-1000? The binary
     label can't distinguish "catches everything" from "catches one easy
     typology" -- this can.
  4. Composition of the near-pure FX tier: which typologies is it made of?

No try/except. If something breaks, it raises.
"""
import os
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
FILES = [os.path.join(DATA, "li_small_full.parquet")]
PATTERNS = os.path.join(DATA, "LI-Small_Patterns.txt")

HUB_CUTOFF = 100
WORKLIST_K = 1000

print("=" * 70)
print("PARSE PATTERNS FILE")
attempts = []          # list of (typology, set_of_accounts)
current_type = None
current_accts = set()
n_rows = 0
with open(PATTERNS, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        m = re.match(r"BEGIN LAUNDERING ATTEMPT - ([A-Z\-]+)", line)
        if m:
            current_type = m.group(1).strip()
            current_accts = set()
            continue
        if line.startswith("END LAUNDERING ATTEMPT"):
            attempts.append((current_type, current_accts))
            current_type = None
            continue
        if current_type is not None:
            parts = line.split(",")
            # Timestamp,FromBank,FromAcct,ToBank,ToAcct,AmtRecv,RecvCur,AmtPaid,PayCur,Format,IsLaundering
            fb, fa, tb, ta = parts[1], parts[2], parts[3], parts[4]
            current_accts.add(f"{int(fb)}:{fa}")
            current_accts.add(f"{int(tb)}:{ta}")
            n_rows += 1

types = sorted({t for t, _ in attempts})
print(f"  attempts: {len(attempts)}   transactions in patterns: {n_rows:,}")
print(f"  typologies: {types}")

acct_types = {}
for t, accs in attempts:
    for a in accs:
        acct_types.setdefault(a, set()).add(t)
pattern_accts = set(acct_types)
print(f"  distinct accounts in any pattern: {len(pattern_accts):,}")
for t in types:
    n_att = sum(1 for tt, _ in attempts if tt == t)
    accs = set().union(*[a for tt, a in attempts if tt == t])
    print(f"    {t:<18} {n_att:>4} attempts   {len(accs):>6,} accounts")

print()
print("=" * 70)
print("COVERAGE vs THE BINARY LABEL")
tx = pd.concat([pd.read_parquet(f) for f in FILES], ignore_index=True)
tx["from_acct"] = tx["From Bank"].astype(str) + ":" + tx["Account"]
tx["to_acct"] = tx["To Bank"].astype(str) + ":" + tx["Account.1"]
all_accts = sorted(set(tx["from_acct"]) | set(tx["to_acct"]))
n = len(all_accts)
idx_of = {a: i for i, a in enumerate(all_accts)}
acct_index = pd.Index(all_accts)
flagged = tx[tx["Is Laundering"] == 1]
pos_accts = set(flagged["from_acct"]) | set(flagged["to_acct"])
label = acct_index.isin(pos_accts).astype(int)

in_graph = pattern_accts & set(all_accts)
print(f"  pattern accounts found in the transaction graph: {len(in_graph):,} "
      f"of {len(pattern_accts):,}")
print(f"  pattern accounts that are label-positive: "
      f"{len(pattern_accts & pos_accts):,}")
print(f"  label-positive accounts NOT in any pattern (background laundering): "
      f"{len(pos_accts - pattern_accts):,} of {len(pos_accts):,} "
      f"({len(pos_accts - pattern_accts)/len(pos_accts):.1%})")

print()
print("=" * 70)
print("REBUILD OPERATIONAL BEST (seed 0) FOR THE RECALL AUDIT")
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
hub_set = set(popularity[popularity > HUB_CUTOFF].index)
kept_cp = counterparty_long[~counterparty_long["counterparty"].isin(hub_set)]
pop_kept = kept_cp.groupby("counterparty")["acct"].nunique()
idf = 1.0 / np.sqrt(1.0 + pop_kept)
rows_ = kept_cp["acct"].map(idx_of).to_numpy()
cols_ = kept_cp["counterparty"].map(idx_of).to_numpy()
M = sp.csr_matrix((idf.reindex(kept_cp["counterparty"]).to_numpy(), (rows_, cols_)), shape=(n, n))
row_norms = np.sqrt(M.multiply(M).sum(axis=1)).A.flatten()
row_norms[row_norms == 0] = 1.0
M = sp.diags(1.0 / row_norms) @ M

cross = tx.loc[tx["Payment Currency"] != tx["Receiving Currency"]]
fx_accts = set(cross["from_acct"]) | set(cross["to_acct"])
has_fx = acct_index.isin(fx_accts).astype(float)

idxs = np.arange(n)
train_idx, test_idx = train_test_split(idxs, test_size=0.4, stratify=label, random_state=0)
y_train, y_test = label[train_idx], label[test_idx]
u1 = M[train_idx].T @ y_train
h1 = np.asarray(M @ u1).flatten()
X = np.column_stack([X_static, h1, has_fx])
sc = StandardScaler()
mdl = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0)
mdl.fit(sc.fit_transform(X[train_idx]), y_train)
scores = mdl.predict_proba(sc.transform(X[test_idx]))[:, 1]
print(f"  AUROC {roc_auc_score(y_test, scores):.4f} (sanity: phase3_transfer_li_full seed 0 was 0.8101)")

test_accts = acct_index[test_idx]
order = np.argsort(-scores)
top_accts = set(test_accts[order[:WORKLIST_K]])
topn_accts = set(test_accts[order[:int(y_test.sum())]])

print()
print("=" * 70)
print(f"PER-TYPOLOGY RECALL (test-set accounts of each typology; worklist = top {WORKLIST_K}, "
      f"also top-npos = {int(y_test.sum())})")
test_set = set(test_accts)
print(f"  {'typology':<18}{'test accts':>11}{'in top1000':>12}{'recall':>9}"
      f"{'in top-npos':>13}{'recall':>9}")
print("  " + "-" * 70)
rows_out = []
for t in types:
    accs = set().union(*[a for tt, a in attempts if tt == t]) & test_set
    if not accs:
        continue
    got = len(accs & top_accts)
    got_n = len(accs & topn_accts)
    print(f"  {t:<18}{len(accs):>11,}{got:>12,}{got/len(accs):>9.1%}"
          f"{got_n:>13,}{got_n/len(accs):>9.1%}")
# background positives (in no pattern) as their own row
bg = (pos_accts - pattern_accts) & test_set
got = len(bg & top_accts)
got_n = len(bg & topn_accts)
print(f"  {'(background)':<18}{len(bg):>11,}{got:>12,}{got/len(bg):>9.1%}"
      f"{got_n:>13,}{got_n/len(bg):>9.1%}")

print()
print("=" * 70)
print("WHAT IS THE FX TIER MADE OF? (top-1000 AND cross-currency, seed 0)")
tier = top_accts & fx_accts
tier_pos = tier & pos_accts
print(f"  tier size {len(tier)}, positive {len(tier_pos)}")
comp = {}
for a in tier_pos:
    for t in acct_types.get(a, {"(background)"}):
        comp[t] = comp.get(t, 0) + 1
for t, c in sorted(comp.items(), key=lambda kv: -kv[1]):
    print(f"    {t:<18}{c:>5}  ({c/len(tier_pos):.1%} of tier positives)")
fp = tier - pos_accts
print(f"  tier false positives: {len(fp)}")
