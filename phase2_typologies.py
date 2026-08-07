"""Phase 2 -- twenty questions over typologies.

PLAN.md: "Needs a typology -> feature matrix that does not exist yet;
building it is the real work, and it must not be built by looking at the
labels."

TYPOLOGY_SIGNATURES below is that matrix. It is written from general AML
typology knowledge (FATF/FinCEN pattern types), BEFORE this script reads
the "Is Laundering" column even once. That ordering is load-bearing, not
decoration: the label is read for the first time after this dict is
defined and after ACCOUNT_FEATURES is computed, in the scoring section at
the bottom.

Boolean per-account features are median-split on the whole population, a
priori (median chosen because it needs no domain threshold, not tuned to
the label). No time windows anywhere -- WHY.md is explicit that window
choice is a threshold in disguise, and none of these features touch a
timestamp.

Two things happen with the catalog:
  1. best_score: for every account, the best Jaccard overlap between its
     own observed features and any typology's signature, scored as a
     ranker exactly like every other Phase 2 feature (AUROC, shuffle
     control, distinct count, precision@k), on the SAME held-out test
     split as prior runs.
  2. per-typology positive rate: group accounts by their single best-
     matching typology, report the label positive rate per group against
     the base rate. This is the actually interpretable output -- "which
     catalogued pattern, if any, concentrates the flagged accounts."
  3. A worked example of the actual TwentyQuestions engine (ported
     unmodified from KOMPOSOS-SEC) narrowing typology candidates against
     one real account's observed features, as a diagnostic/investigator
     tool -- not a bulk scoring device, which is why (1) and (2) exist
     separately.

No try/except. If something breaks, it raises.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from twenty_questions import TwentyQuestions, entropy

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FILES = [os.path.join(DATA, "hi_small_0.parquet"),
         os.path.join(DATA, "hi_small_1.parquet")]

SEED = 0

# ---------------------------------------------------------------------------
# TYPOLOGY CATALOG -- written before the label column is read. See docstring.
# ---------------------------------------------------------------------------
TYPOLOGY_SIGNATURES = {
    "round_robin_layering": {"has_2cycle", "has_3cycle", "multi_bank"},
    "mule_fan_out_network": {"high_fan_out", "low_fan_in", "high_num_tx"},
    "collector_fan_in":     {"high_fan_in", "low_fan_out"},
    "pass_through_funnel":  {"high_fan_in", "high_fan_out", "balanced_flow", "high_num_tx"},
    "structuring_high_freq": {"high_num_tx", "low_volume", "single_bank"},
    "cross_bank_layering":  {"multi_bank", "high_num_tx", "lopsided_flow"},
    "concentrated_relationship": {"low_fan_out", "low_fan_in", "high_volume"},
    "dormant_low_activity": {"low_num_tx", "low_volume", "single_bank"},
}
ALL_FEATURES = sorted({f for sig in TYPOLOGY_SIGNATURES.values() for f in sig})

print("=" * 70)
print("TYPOLOGY CATALOG (fixed before the label column is read)")
for name, sig in TYPOLOGY_SIGNATURES.items():
    print(f"  {name:<26} {sorted(sig)}")
print(f"  {len(TYPOLOGY_SIGNATURES)} typologies, {len(ALL_FEATURES)} distinct features")
print(f"  entropy floor: log2({len(TYPOLOGY_SIGNATURES)}) = "
      f"{entropy(len(TYPOLOGY_SIGNATURES)):.3f} bits")

# ---------------------------------------------------------------------------
# BUILD PER-ACCOUNT STRUCTURE (no label read yet)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("LOAD + BUILD PER-ACCOUNT STRUCTURE")
tx = pd.concat([pd.read_parquet(f) for f in FILES], ignore_index=True)
print(f"  combined: {len(tx):,} rows")

tx["from_acct"] = tx["From Bank"].astype(str) + ":" + tx["Account"]
tx["to_acct"] = tx["To Bank"].astype(str) + ":" + tx["Account.1"]

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

all_accts = set(tx["from_acct"]) | set(tx["to_acct"])
acc = pd.DataFrame(index=sorted(all_accts))
acc.index.name = "acct"
acc["out_degree"] = out_degree.reindex(acc.index).fillna(0)
acc["in_degree"] = in_degree.reindex(acc.index).fillna(0)
acc["amount_out"] = out_amt.reindex(acc.index).fillna(0.0)
acc["amount_in"] = in_amt.reindex(acc.index).fillna(0.0)
acc["num_transactions"] = (out_cnt.reindex(acc.index).fillna(0)
                            + in_cnt.reindex(acc.index).fillna(0))
acc["distinct_banks"] = distinct_banks.reindex(acc.index).fillna(0)
acc["reciprocal_count"] = reciprocal_count.reindex(acc.index).fillna(0)
acc["tri3_count"] = tri3_count.reindex(acc.index).fillna(0)
acc["in_ratio"] = acc["amount_in"] / (acc["amount_in"] + acc["amount_out"])
acc["total_volume"] = acc["amount_in"] + acc["amount_out"]
n_accts = len(acc)

# --- boolean features, median split, a priori -------------------------
med_fan_out = acc["out_degree"].median()
med_fan_in = acc["in_degree"].median()
med_num_tx = acc["num_transactions"].median()
med_volume = acc["total_volume"].median()
print(f"  medians used for split: out_degree={med_fan_out}, in_degree={med_fan_in}, "
      f"num_transactions={med_num_tx}, total_volume={med_volume:.2f}")

acc["has_2cycle"] = acc["reciprocal_count"] > 0
acc["has_3cycle"] = acc["tri3_count"] > 0
acc["high_fan_out"] = acc["out_degree"] > med_fan_out
acc["low_fan_out"] = ~acc["high_fan_out"]
acc["high_fan_in"] = acc["in_degree"] > med_fan_in
acc["low_fan_in"] = ~acc["high_fan_in"]
acc["high_num_tx"] = acc["num_transactions"] > med_num_tx
acc["low_num_tx"] = ~acc["high_num_tx"]
acc["multi_bank"] = acc["distinct_banks"] > 1
acc["single_bank"] = ~acc["multi_bank"]
acc["balanced_flow"] = acc["in_ratio"].between(0.4, 0.6)
acc["lopsided_flow"] = ~acc["balanced_flow"]
acc["high_volume"] = acc["total_volume"] > med_volume
acc["low_volume"] = ~acc["high_volume"]

# ---------------------------------------------------------------------------
# BEST-MATCH SCORING (vectorized Jaccard against each typology signature)
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("MATCH EACH ACCOUNT AGAINST THE CATALOG")
typ_names = list(TYPOLOGY_SIGNATURES)
F = acc[ALL_FEATURES].to_numpy(dtype=bool).astype(np.int32)             # (n, f)
S = np.array([[1 if feat in TYPOLOGY_SIGNATURES[t] else 0 for feat in ALL_FEATURES]
              for t in typ_names], dtype=np.int32)                      # (t, f)

intersection = F @ S.T                                                  # (n, t)
acc_count = F.sum(axis=1, keepdims=True)                                # (n, 1)
sig_count = S.sum(axis=1, keepdims=True).T                              # (1, t)
union = acc_count + sig_count - intersection
jaccard = intersection / union

best_idx = jaccard.argmax(axis=1)
best_score = jaccard.max(axis=1)
acc["best_typology"] = [typ_names[i] for i in best_idx]
acc["best_score"] = best_score

print(f"  scored {n_accts:,} accounts against {len(typ_names)} typologies")
print(f"  best_score distinct values: {pd.Series(best_score).nunique()}")

# ---------------------------------------------------------------------------
# LABEL READ FOR THE FIRST TIME
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("LABEL READ (first time in this script)")
flagged = tx[tx["Is Laundering"] == 1]
pos_accts = set(flagged["from_acct"]) | set(flagged["to_acct"])
acc["label"] = acc.index.isin(pos_accts).astype(int)
n_pos = int(acc["label"].sum())
print(f"  positive accounts: {n_pos:,} = {n_pos/n_accts:.4%}")

print()
print("=" * 70)
print("PER-TYPOLOGY POSITIVE RATE (best-match assignment)")
grp = acc.groupby("best_typology")["label"].agg(["count", "sum"])
grp["rate"] = grp["sum"] / grp["count"]
grp = grp.sort_values("rate", ascending=False)
print(f"  {'typology':<26}{'n':>10}{'positive':>10}{'rate':>10}{'lift':>8}")
base_rate = n_pos / n_accts
for name, row in grp.iterrows():
    lift = row["rate"] / base_rate if base_rate > 0 else float("nan")
    print(f"  {name:<26}{int(row['count']):>10,}{int(row['sum']):>10,}"
          f"{row['rate']:>10.4%}{lift:>8.2f}x")
print(f"  (overall base rate: {base_rate:.4%})")

# ---------------------------------------------------------------------------
# SCORE best_score AS A RANKER, SAME PROTOCOL AS EVERY OTHER PHASE 2 FEATURE
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("SCORE best_score ON HELD-OUT TEST SET (same split as prior runs)")
y = acc["label"].to_numpy()
idx = np.arange(n_accts)
train_idx, test_idx = train_test_split(idx, test_size=0.4, stratify=y, random_state=SEED)
y_test = y[test_idx]
n_pos_test = int(y_test.sum())

rng = np.random.default_rng(SEED)
y_test_shuffled = y_test.copy()
rng.shuffle(y_test_shuffled)


def precision_at_k(scores, labels, k):
    threshold = np.partition(scores, -k)[-k]
    selected = scores >= threshold
    n_selected = int(selected.sum())
    n_hit = int(labels[selected].sum())
    return n_hit, n_selected


scores = acc["best_score"].to_numpy()[test_idx]
distinct = int(pd.Series(scores).nunique())
auroc = roc_auc_score(y_test, scores)
auroc_shuf = roc_auc_score(y_test_shuffled, scores)
hits, sel = precision_at_k(scores, y_test, n_pos_test)
prec = hits / sel

header = (f"{'feature':<18}{'AUROC':>10}{'AUROC(shuf)':>13}"
          f"{'distinct':>10}{'prec@k':>10}{'k':>8}{'hits':>7}{'sel':>7}")
print(header)
print("-" * len(header))
print(f"{'typology_best':<18}{auroc:>10.4f}{auroc_shuf:>13.4f}"
      f"{distinct:>10,}{prec:>10.4f}{n_pos_test:>8,}{hits:>7,}{sel:>7,}")

print()
print("  Prior best: combined_all (logreg) 0.7759 AUROC / 0.0861 prec@k")
print("              reciprocal_count      0.5390 AUROC / 0.0857 prec@k")

# ---------------------------------------------------------------------------
# WORKED EXAMPLE: the actual TwentyQuestions engine narrowing typologies
# against one real account's observed features
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("WORKED EXAMPLE: TwentyQuestions engine on one real flagged account")
example_acct = acc[(acc["label"] == 1) & (acc["reciprocal_count"] > 0)].index
if len(example_acct) == 0:
    example_acct = acc[acc["label"] == 1].index
example = example_acct[0]
observed = {f for f in ALL_FEATURES if acc.loc[example, f]}
print(f"  account: {example}")
print(f"  observed features: {sorted(observed)}")

game = TwentyQuestions(TYPOLOGY_SIGNATURES)
print(f"  starting bits: {game.bits_remaining():.3f} "
      f"({len(game.candidates)} candidate typologies)")
asked = set()
for step in range(6):
    if len(game.candidates) <= 1:
        break
    q = game.best_question(exclude=asked)
    if q is None:
        break
    answer = q.technique in observed
    asked.add(q.technique)
    survivors = game.answer(q.technique, answer)
    print(f"  Q{step+1}: {q.technique}?  -> {answer}   "
          f"gain {q.information_gain:.3f} bits   {survivors} candidates left "
          f"({game.bits_remaining():.3f} bits)")
print(f"  final candidates: {sorted(game.candidates)}")
