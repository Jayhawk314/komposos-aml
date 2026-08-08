"""Phase 6 -- FROZEN transfer test on HI-Large: the ladder's last rung.

~180M transactions. The frozen recipe's MATH is unchanged from every prior
transfer run (same features, same hub rule >100, same split protocol, same
model). The IMPLEMENTATION is rebuilt for memory: integer account codes +
np.bincount aggregations replace string-keyed pandas groupbys, and the input
is a slim parquet (data/hi_large_slim.parquet) carrying only the columns the
recipe reads. Feature-value equivalence notes:

  - degree/in_degree/out_degree: nunique counterparties from deduplicated
    directed pairs (self-loops included, as tx-frame groupby.nunique did).
  - distinct_banks: original build gave each account, per transaction, both
    endpoint banks; the set over all transactions equals {own bank} union
    {counterparty banks} -- computed here from deduplicated pairs + the
    bank prefix of the account key. Identical value, fraction of the memory.
  - triangles: same enumeration as always (no hub guard -- HI-Medium ran
    unguarded through a 58K hub). The wedge count is computed and printed
    BEFORE the merge; if it exceeds 800M rows the script raises with the
    measured number -- an honest infeasibility result, not a silent hang.

If it dies on memory, the death point is the finding: the laptop's ceiling.

No try/except. If something breaks, it raises.
"""
import gc
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
SLIM = os.path.join(DATA, "hi_large_slim.parquet")

SEEDS = [0, 1, 2, 3, 4]
K_VALUES = [100, 500, 1000]
HUB_CUTOFF = 100
WORKLIST_K = 1000

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

print("=" * 70)
print("LOAD (pyarrow; 180M-row string columns never materialize in pandas)")
t = pq.read_table(SLIM, columns=["from_acct", "to_acct", "Amount Received",
                                  "Amount Paid", "Payment Currency",
                                  "Receiving Currency", "Is Laundering"])
n_rows = t.num_rows
print(f"  rows: {n_rows:,}")

print("  encoding accounts as integer codes (arrow unique + index_in)...")
both = pa.chunked_array(t["from_acct"].chunks + t["to_acct"].chunks)
uniq = pc.unique(both).sort()
del both
n = len(uniq)
cats_pd = uniq.to_pandas()  # ~12M short strings, needed for hub names + banks
from_c = pc.index_in(t["from_acct"], value_set=uniq).combine_chunks().to_numpy(
    zero_copy_only=False).astype(np.int32)
to_c = pc.index_in(t["to_acct"], value_set=uniq).combine_chunks().to_numpy(
    zero_copy_only=False).astype(np.int32)
print(f"  accounts: {n:,}")

is_pos_tx = pc.equal(t["Is Laundering"], 1).combine_chunks().to_numpy(
    zero_copy_only=False)
label = np.zeros(n, dtype=np.int8)
label[from_c[is_pos_tx]] = 1
label[to_c[is_pos_tx]] = 1
n_pos = int(label.sum())
print(f"  positive accounts: {n_pos:,} = {n_pos/n:.4%}")

print()
print("=" * 70)
print("FEATURES (bincount implementations of the identical definitions)")
amt_paid = t["Amount Paid"].combine_chunks().to_numpy(zero_copy_only=False)
amt_recv = t["Amount Received"].combine_chunks().to_numpy(zero_copy_only=False)
amount_out = np.bincount(from_c, weights=amt_paid, minlength=n)
amount_in = np.bincount(to_c, weights=amt_recv, minlength=n)
out_cnt = np.bincount(from_c, minlength=n).astype(np.float64)
in_cnt = np.bincount(to_c, minlength=n).astype(np.float64)
num_transactions = out_cnt + in_cnt

fx_mask = pc.not_equal(t["Payment Currency"],
                       t["Receiving Currency"]).combine_chunks().to_numpy(
    zero_copy_only=False)
has_fx = np.zeros(n, dtype=np.float64)
has_fx[from_c[fx_mask]] = 1.0
has_fx[to_c[fx_mask]] = 1.0
print(f"  cross-currency rows: {int(fx_mask.sum()):,} ({fx_mask.mean():.4%}); "
      f"accounts with any: {int(has_fx.sum()):,} ({has_fx.mean():.2%})")
del t, uniq, amt_paid, amt_recv, is_pos_tx, fx_mask
gc.collect()

print("  deduplicating directed pairs...")
pair_key = from_c.astype(np.int64) * n + to_c
pair_key = np.unique(pair_key)
e_from = (pair_key // n).astype(np.int32)
e_to = (pair_key % n).astype(np.int32)
del pair_key, from_c, to_c
gc.collect()
print(f"  unique directed pairs (self-loops incl.): {len(e_from):,}")

out_degree = np.bincount(e_from, minlength=n).astype(np.float64)
in_degree = np.bincount(e_to, minlength=n).astype(np.float64)
degree = out_degree + in_degree

# counterparty pairs (both directions, deduped) -> popularity, CF, banks
ck = np.concatenate([e_from.astype(np.int64) * n + e_to,
                     e_to.astype(np.int64) * n + e_from])
ck = np.unique(ck)
cp_acct = (ck // n).astype(np.int32)
cp_cp = (ck % n).astype(np.int32)
del ck
gc.collect()
popularity = np.bincount(cp_cp, minlength=n)
hubs = np.where(popularity > HUB_CUTOFF)[0]
hub_names = [cats_pd.iloc[h] for h in hubs]
print(f"  frozen hub rule (>{HUB_CUTOFF}) excludes {len(hubs)} counterparties "
      f"(top: {sorted(((cats_pd.iloc[h], int(popularity[h])) for h in hubs), key=lambda kv: -kv[1])[:5]})")

# distinct_banks: {own bank} union {banks of counterparties}
bank_of = cats_pd.str.split(":", n=1).str[0].to_numpy()
bank_codes = pd.factorize(bank_of)[0].astype(np.int64)
n_banks = int(bank_codes.max()) + 1
bk = np.unique(cp_acct.astype(np.int64) * n_banks + bank_codes[cp_cp])
distinct_banks_cp = np.bincount((bk // n_banks).astype(np.int32), minlength=n).astype(np.float64)
own_bank_seen = np.zeros(n, dtype=bool)
same_bank = bank_codes[cp_acct] == bank_codes[cp_cp]
own_bank_seen[cp_acct[same_bank]] = True
distinct_banks = distinct_banks_cp + (~own_bank_seen & (degree > 0))
del bk, same_bank, distinct_banks_cp, own_bank_seen, bank_of
gc.collect()

# reciprocal + triangles on non-self-loop directed edges
nsl = e_from != e_to
ef, et = e_from[nsl], e_to[nsl]
print(f"  non-self-loop edges: {len(ef):,}")
edge_df = pd.DataFrame({"u": ef, "v": et})
rec = edge_df.merge(edge_df, left_on=["u", "v"], right_on=["v", "u"])
reciprocal_count = rec.groupby("u_x")["v_x"].nunique()
reciprocal = np.zeros(n)
reciprocal[reciprocal_count.index.to_numpy()] = reciprocal_count.to_numpy()
del rec, reciprocal_count
gc.collect()

wedge_est = int(np.sum(np.bincount(et, minlength=n).astype(np.int64) *
                        np.bincount(ef, minlength=n).astype(np.int64)))
print(f"  wedge (2-hop path) count for triangle build: {wedge_est:,}")
if wedge_est > 800_000_000:
    raise RuntimeError(
        f"Triangle enumeration infeasible at this scale: {wedge_est:,} wedges "
        f"(> 800M, ~10+ GB as a frame). Measured result -- record in FINDINGS.md.")
hop2 = edge_df.merge(edge_df, left_on="v", right_on="u", suffixes=("1", "2"))
hop2 = hop2.rename(columns={"u1": "u", "v1": "m", "v2": "w"})[["u", "m", "w"]]
hop2 = hop2[hop2["u"] != hop2["w"]]
close_key = hop2["w"].to_numpy().astype(np.int64) * n + hop2["u"].to_numpy()
edge_key = np.sort(ef.astype(np.int64) * n + et)
closed = hop2[np.isin(close_key, edge_key)]
del hop2, close_key, edge_key
gc.collect()
tri = closed.drop_duplicates()
tri3 = np.zeros(n)
for col in ["u", "m", "w"]:
    cnt = tri.groupby(col).size()
    tri3[cnt.index.to_numpy()] += cnt.to_numpy()
del closed, tri, edge_df
gc.collect()
print(f"  triangle build complete")

X_static = np.column_stack([degree, in_degree, out_degree, amount_out, amount_in,
                            num_transactions, distinct_banks, reciprocal, tri3])

print("  building CF matrix (hub-excluded, sqrt-IDF, row-normalized)...")
keep = ~np.isin(cp_cp, hubs)
r, c = cp_acct[keep], cp_cp[keep]
pop_kept = np.bincount(c, minlength=n)
idf_vals = 1.0 / np.sqrt(1.0 + pop_kept[c])
M = sp.csr_matrix((idf_vals, (r, c)), shape=(n, n))
row_norms = np.sqrt(M.multiply(M).sum(axis=1)).A.flatten()
row_norms[row_norms == 0] = 1.0
M = sp.diags(1.0 / row_norms) @ M
del keep, r, c, idf_vals, cp_acct, cp_cp, row_norms
gc.collect()
print(f"  CF matrix: {M.nnz:,} nonzero")


def precision_at_k(scores, labels, k):
    threshold = np.partition(scores, -k)[-k]
    selected = scores >= threshold
    return int(labels[selected].sum()), int(selected.sum())


idxs = np.arange(n)
results = []
print()
print("=" * 70)
print("5-SEED RUNS -- frozen recipe, HI-Large")
for seed in SEEDS:
    tr, te = train_test_split(idxs, test_size=0.4, stratify=label, random_state=seed)
    ytr, yte = label[tr].astype(np.int32), label[te].astype(np.int32)
    npt = int(yte.sum())
    rng = np.random.default_rng(seed)
    ysh = yte.copy()
    rng.shuffle(ysh)

    u1 = M[tr].T @ ytr
    h1 = np.asarray(M @ u1).flatten()
    X = np.column_stack([X_static, h1, has_fx])
    sc = StandardScaler()
    mdl = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    mdl.fit(sc.fit_transform(X[tr]), ytr)
    s = mdl.predict_proba(sc.transform(X[te]))[:, 1]

    row = {"seed": seed, "auroc": roc_auc_score(yte, s),
           "auroc_shuf": roc_auc_score(ysh, s),
           "distinct": int(pd.Series(s).nunique())}
    for k in K_VALUES + [npt]:
        key = k if k in K_VALUES else "npos"
        hits, sel = precision_at_k(s, yte, k)
        row[f"p@{key}"] = hits / sel
        row[f"sel@{key}"] = sel

    order = np.argsort(-s)[:WORKLIST_K]
    top_fx = has_fx[te[order]] > 0
    n_fx = int(top_fx.sum())
    fx_pos = int(yte[order][top_fx].sum())
    rest = WORKLIST_K - n_fx
    row["fx_tier_n"] = n_fx
    row["fx_tier_rate"] = fx_pos / n_fx if n_fx else float("nan")
    row["rest_rate"] = int(yte[order][~top_fx].sum()) / rest if rest else float("nan")
    results.append(row)
    print(f"  seed {seed}: AUROC {row['auroc']:.4f} (shuf {row['auroc_shuf']:.4f})  "
          f"distinct {row['distinct']:,}  p@100 {row['p@100']:.4f} (sel {row['sel@100']})  "
          f"p@500 {row['p@500']:.4f}  p@1000 {row['p@1000']:.4f}  "
          f"p@npos({npt}) {row['p@npos']:.4f}")
    print(f"          fx tier {fx_pos}/{n_fx} = {row['fx_tier_rate']:.1%}  |  "
          f"rest {row['rest_rate']:.1%}")
    del X, s, u1, h1
    gc.collect()

res = pd.DataFrame(results)
base = n_pos / n
print()
print("=" * 70)
print("LADDER VERDICT (mean across 5 seeds)")
print(f"  {'metric':<14}{'HI-Large':>12}{'HI-Medium':>12}{'HI-Small':>12}{'LI-Small':>12}")
print("  " + "-" * 64)
ref = {"auroc": (0.8736, 0.8991, 0.8075), "p@100": (0.7900, 0.8740, 0.4700),
       "p@500": (0.8176, 0.7176, 0.3352), "p@1000": (0.7682, 0.5912, 0.3372),
       "fx_tier_rate": (0.9958, 0.9740, 0.8654)}
for metric, (med, small, li) in ref.items():
    print(f"  {metric:<14}{res[metric].mean():>12.4f}{med:>12.4f}{small:>12.4f}{li:>12.4f}")
print(f"  {'shuffle':<14}{res['auroc_shuf'].mean():>12.4f}")
print(f"  base rate: {base:.4%}   lift@100: {res['p@100'].mean()/base:.1f}x")
