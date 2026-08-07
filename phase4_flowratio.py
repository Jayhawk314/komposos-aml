"""Phase 4 -- flow-ratio feature, second attempt at the STACK blind spot.

Butterfly counting (phase4_butterfly.py) failed its criterion: the generator's
BIPARTITE/STACK attempts are DISJOINT single edges / 2-hop chains with fresh
counterparties -- no shared-neighbor density exists to count. Verified by
reading the attempts directly.

Remaining label-free hook: STACK middles are pass-throughs (receive X,
forward ~0.85X). Feature: flow_ratio = min(amount_in, amount_out) /
max(amount_in, amount_out), continuous, threshold-free, 0 when either side
is 0. BIPARTITE members have no such hook (single one-way edges) -- stated in
advance: this targets STACK; BIPARTITE likely stays at its floor.

SUCCESS CRITERION, decided in advance: STACK recall@npos improves
substantially; overall metrics and other typologies do not degrade (5 seeds).

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
FILES = [os.path.join(DATA, "hi_small_0.parquet"),
         os.path.join(DATA, "hi_small_1.parquet")]
PATTERNS = os.path.join(DATA, "HI-Small_Patterns.txt")

SEEDS = [0, 1, 2, 3, 4]
K_VALUES = [100, 500, 1000]
HUB_CUTOFF = 100

print("=" * 70)
print("LOAD")
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
print(f"  rows {len(tx):,}   accounts {n:,}   positive {int(label.sum()):,}")

print()
print("=" * 70)
print("FLOW RATIO (labels untouched)")
counterparty_long = pd.concat([
    tx[["from_acct", "to_acct"]].rename(columns={"from_acct": "acct", "to_acct": "counterparty"}),
    tx[["to_acct", "from_acct"]].rename(columns={"to_acct": "acct", "from_acct": "counterparty"}),
], ignore_index=True).drop_duplicates()
popularity = counterparty_long.groupby("counterparty")["acct"].nunique()
hub_set = set(popularity[popularity > HUB_CUTOFF].index)

edges_d = tx.loc[tx["from_acct"] != tx["to_acct"], ["from_acct", "to_acct"]].drop_duplicates()

amt_in_s = tx.groupby("to_acct")["Amount Received"].sum().reindex(all_accts).fillna(0.0)
amt_out_s = tx.groupby("from_acct")["Amount Paid"].sum().reindex(all_accts).fillna(0.0)
lo = np.minimum(amt_in_s.to_numpy(), amt_out_s.to_numpy())
hi = np.maximum(amt_in_s.to_numpy(), amt_out_s.to_numpy())
flow_ratio = np.where(hi > 0, lo / hi, 0.0)
nz = int((flow_ratio > 0).sum())
print(f"  accounts with two-sided flow (ratio > 0): {nz:,} ({nz/n:.2%})")
print(f"  flow_ratio distribution: median {np.median(flow_ratio[flow_ratio>0]):.4f}, "
      f"p90 {np.quantile(flow_ratio[flow_ratio>0], 0.9):.4f}")

print()
print("=" * 70)
print("STANDARD FEATURE BUILD (as every prior script)")
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
reciprocal = edges_d.merge(edges_d, left_on=["from_acct", "to_acct"],
                            right_on=["to_acct", "from_acct"])
reciprocal_count = reciprocal.groupby("from_acct_x")["to_acct_x"].nunique()
edges2 = edges_d.rename(columns={"from_acct": "u", "to_acct": "v"})
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

print()
print("=" * 70)
print("PARSE PATTERNS (for the per-typology recall verdict)")
attempts = []
current_type = None
current_accts = set()
with open(PATTERNS, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        m = re.match(r"BEGIN LAUNDERING ATTEMPT - ([A-Z\-]+)", line)
        if m:
            current_type = m.group(1)
            current_accts = set()
            continue
        if line.startswith("END LAUNDERING ATTEMPT"):
            attempts.append((current_type, current_accts))
            current_type = None
            continue
        if current_type is not None:
            p = line.split(",")
            current_accts.add(f"{int(p[1])}:{p[2]}")
            current_accts.add(f"{int(p[3])}:{p[4]}")
types = sorted({t for t, _ in attempts})
type_accts = {t: set().union(*[a for tt, a in attempts if tt == t]) for t in types}
pattern_accts = set().union(*type_accts.values())

# descriptive: median flow_ratio per typology (whole population, labels only for reporting)
fr_s = pd.Series(flow_ratio, index=all_accts)
print(f"  {'typology':<18}{'accounts':>9}{'median flow_ratio':>19}")
for t in types:
    accs = list(type_accts[t])
    print(f"  {t:<18}{len(accs):>9,}{fr_s.reindex(accs).median():>19.4f}")


def precision_at_k(scores, labels, k):
    threshold = np.partition(scores, -k)[-k]
    selected = scores >= threshold
    return int(labels[selected].sum()), int(selected.sum())


def fit(feats, tr, te, ytr, seed):
    X = np.column_stack(feats)
    sc = StandardScaler()
    mdl = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    mdl.fit(sc.fit_transform(X[tr]), ytr)
    return mdl.predict_proba(sc.transform(X[te]))[:, 1]


idxs = np.arange(n)
results = {"base": [], "with_flow": []}
recalls = {}
print()
print("=" * 70)
print("5-SEED RUNS: operational best without vs with butterfly features")
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
        "base": fit([X_static, h1, has_fx], tr, te, ytr, seed),
        "with_flow": fit([X_static, h1, has_fx, flow_ratio], tr, te, ytr, seed),
    }
    for name, s in variants.items():
        row = {"seed": seed, "auroc": roc_auc_score(yte, s),
               "auroc_shuf": roc_auc_score(ysh, s),
               "distinct": int(pd.Series(s).nunique())}
        for k in K_VALUES + [npt]:
            key = k if k in K_VALUES else "npos"
            hits, sel = precision_at_k(s, yte, k)
            row[f"p@{key}"] = hits / sel
        results[name].append(row)
        print(f"  seed {seed} {name:<8} AUROC {row['auroc']:.4f} (shuf {row['auroc_shuf']:.4f})  "
              f"p@100 {row['p@100']:.4f}  p@500 {row['p@500']:.4f}  "
              f"p@1000 {row['p@1000']:.4f}  p@npos {row['p@npos']:.4f}")
        if seed == 0:
            test_accts = acct_index[te]
            order = np.argsort(-s)
            topn = set(test_accts[order[:npt]])
            top1000 = set(test_accts[order[:1000]])
            test_set = set(test_accts)
            recalls[name] = {}
            for t in types:
                accs = type_accts[t] & test_set
                recalls[name][t] = (len(accs & top1000) / len(accs),
                                    len(accs & topn) / len(accs), len(accs))
            bg = (pos_accts - pattern_accts) & test_set
            recalls[name]["(background)"] = (len(bg & top1000) / len(bg),
                                             len(bg & topn) / len(bg), len(bg))

print()
print("=" * 70)
print("VERDICT 1 -- overall metrics must not degrade (mean of 5 seeds)")
ra, rb = pd.DataFrame(results["base"]), pd.DataFrame(results["with_flow"])
print(f"  {'metric':<12}{'base':>10}{'with_flow':>10}{'delta':>10}")
print("  " + "-" * 42)
for metric in ["auroc", "auroc_shuf", "p@100", "p@500", "p@1000", "p@npos"]:
    a, b = ra[metric].mean(), rb[metric].mean()
    print(f"  {metric:<12}{a:>10.4f}{b:>10.4f}{b - a:>+10.4f}")

print()
print("VERDICT 2 -- the targets: per-typology recall, seed 0")
print(f"  {'typology':<18}{'n':>6}{'top1000 base':>14}{'top1000 fl':>12}"
      f"{'npos base':>11}{'npos fl':>9}")
print("  " + "-" * 72)
for t in types + ["(background)"]:
    b1000, bnpos, cnt = recalls["base"][t]
    w1000, wnpos, _ = recalls["with_flow"][t]
    mark = "  <-- target" if t in ("STACK", "BIPARTITE") else ""
    print(f"  {t:<18}{cnt:>6,}{b1000:>13.1%}{w1000:>12.1%}"
          f"{bnpos:>11.1%}{wnpos:>9.1%}{mark}")
