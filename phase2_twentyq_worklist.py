"""Phase 2 -- twenty questions over the operational worklist.

PLAN.md phase-2 idea #2, finally applied to a real candidate set (CLAUDE.md:
"not yet applied beyond the typology-catalog worked example"). twenty_questions.py
is used UNMODIFIED -- it takes any Dict[str, Set[str]] candidate -> features
population.

The population: the top-1000 worklist from the operational best model
(static + hub-excluded CF + has_fx flag, seed-0 split -- the same list an
investigator would actually be handed, ~59% positive).

The features: ONLY natural zero/nonzero binary facts, no tuned thresholds,
none built by looking at labels -- each is one bounded graph/ledger query:
sends/receives at all, multi-bank, reciprocal edge, 3-cycle membership,
self-loop, cross-currency, multi-currency, touches one of the 15 structural
hubs, uses payment format X, uses currency Y.

What is measured, honestly:
  1. DISTINCT SIGNATURES among the 1000 -- the hard ceiling on identification.
     If suspects are structurally identical, no questioning can separate them.
  2. Mean questions to uniquely identify a suspect (information strategy vs
     the ask-the-most-common-fact baseline), vs the log2(N) theoretical floor.
  3. DESCRIPTIVE ONLY: per top question, the positive rate on the yes and no
     branches within the worklist. This is reporting, not a detector claim --
     the game identifies candidates, it does not classify guilt.

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

from twenty_questions import TwentyQuestions, entropy, evaluate

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FILES = [os.path.join(DATA, "hi_small_0.parquet"),
         os.path.join(DATA, "hi_small_1.parquet")]

HUB_CUTOFF = 100
WORKLIST_K = 1000

print("=" * 70)
print("LOAD + REBUILD THE OPERATIONAL BEST MODEL (seed-0 split, as everywhere)")
tx = pd.concat([pd.read_parquet(f) for f in FILES], ignore_index=True)
tx["from_acct"] = tx["From Bank"].astype(str) + ":" + tx["Account"]
tx["to_acct"] = tx["To Bank"].astype(str) + ":" + tx["Account.1"]
print(f"  combined: {len(tx):,} rows")

all_accts = sorted(set(tx["from_acct"]) | set(tx["to_acct"]))
n = len(all_accts)
idx_of = {a: i for i, a in enumerate(all_accts)}
acct_index = pd.Index(all_accts)

flagged = tx[tx["Is Laundering"] == 1]
pos_accts = set(flagged["from_acct"]) | set(flagged["to_acct"])
label = acct_index.isin(pos_accts).astype(int)

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
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X[train_idx])
X_test_s = scaler.transform(X[test_idx])
model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0)
model.fit(X_train_s, y_train)
scores = model.predict_proba(X_test_s)[:, 1]
print(f"  model rebuilt: AUROC {roc_auc_score(y_test, scores):.4f} "
      f"(must match phase2_invariants.py seed 0 with_fxflag: 0.9012)")

order = np.argsort(-scores)
worklist_local = order[:WORKLIST_K]
worklist_accts = [all_accts[test_idx[i]] for i in worklist_local]
worklist_labels = {all_accts[test_idx[i]]: int(y_test[i]) for i in worklist_local}
n_pos_wl = sum(worklist_labels.values())
print(f"  worklist: top {WORKLIST_K}, {n_pos_wl} positive = {n_pos_wl/WORKLIST_K:.2%}")

print()
print("=" * 70)
print("BINARY FEATURE SETS (natural zero/nonzero facts only, no thresholds)")
self_loop_accts = set(tx.loc[tx["from_acct"] == tx["to_acct"], "from_acct"])
hub_touchers = set(counterparty_long.loc[
    counterparty_long["counterparty"].isin(hub_set), "acct"])

fmt_long = pd.concat([
    tx[["from_acct", "Payment Format"]].rename(columns={"from_acct": "acct"}),
    tx[["to_acct", "Payment Format"]].rename(columns={"to_acct": "acct"}),
], ignore_index=True).drop_duplicates()
fmt_by_acct = fmt_long.groupby("acct")["Payment Format"].agg(set)

cur_long = pd.concat([
    tx[["from_acct", "Payment Currency"]].rename(
        columns={"from_acct": "acct", "Payment Currency": "cur"}),
    tx[["to_acct", "Receiving Currency"]].rename(
        columns={"to_acct": "acct", "Receiving Currency": "cur"}),
], ignore_index=True).drop_duplicates()
cur_by_acct = cur_long.groupby("acct")["cur"].agg(set)

deg = acc["degree"]
population = {}
for a in worklist_accts:
    feats = set()
    if acc.at[a, "in_degree"] > 0:
        feats.add("receives")
    if acc.at[a, "out_degree"] > 0:
        feats.add("sends")
    if acc.at[a, "distinct_banks"] > 1:
        feats.add("multi_bank")
    if acc.at[a, "reciprocal_count"] > 0:
        feats.add("reciprocal_edge")
    if acc.at[a, "tri3_count"] > 0:
        feats.add("in_3cycle")
    if a in self_loop_accts:
        feats.add("self_loop")
    if a in fx_accts:
        feats.add("cross_currency")
    if a in hub_touchers:
        feats.add("touches_hub")
    fmts = fmt_by_acct.get(a, set())
    for f in fmts:
        feats.add(f"format:{f}")
    curs = cur_by_acct.get(a, set())
    for c in curs:
        feats.add(f"currency:{c}")
    if len(curs) > 1:
        feats.add("multi_currency")
    population[a] = feats

all_feats = set().union(*population.values())
print(f"  candidates: {len(population):,}   distinct binary facts in play: {len(all_feats)}")

signatures = {}
for a, f in population.items():
    signatures.setdefault(frozenset(f), []).append(a)
n_sig = len(signatures)
biggest = max(signatures.values(), key=len)
print(f"  DISTINCT SIGNATURES: {n_sig} of {len(population)} "
      f"-- the hard ceiling on identification")
biggest_sig = next(s for s, accs in signatures.items() if accs is biggest)
print(f"  largest identical-signature block: {len(biggest)} accounts "
      f"({sum(worklist_labels[a] for a in biggest)} positive) with facts: "
      f"{sorted(biggest_sig)}")

print()
print("=" * 70)
print("THE GAME (twenty_questions.py, unmodified)")
print(f"  theoretical floor: log2({len(population)}) = "
      f"{entropy(len(population)):.1f} perfect questions")
results = evaluate(population, sample=60, seed=0)
for strategy, stats in results.items():
    print(f"  {strategy:<12} mean questions {stats['mean_questions']:.1f}   "
          f"identified within 20: {stats['identified_within_20']:.0%}")

print()
print("=" * 70)
print("EXAMPLE GAME -- the checklist an investigator would run, in order")
print("(answers assumed YES to walk the deepest branch; per-question worklist")
print(" positive rates on each branch are DESCRIPTIVE, not a detector claim)")
game = TwentyQuestions(population)
asked = set()
for step in range(10):
    q = game.best_question(exclude=asked)
    if q is None:
        break
    users = game.by_technique[q.technique]
    cand = game.candidates
    yes_accts = users & cand
    no_accts = cand - users
    yes_pos = sum(worklist_labels[a] for a in yes_accts)
    no_pos = sum(worklist_labels[a] for a in no_accts)
    print(f"  {step+1:>2}. {game.bits_remaining():5.1f} bits left | {q.technique:<28} "
          f"gain {q.information_gain:.3f} | "
          f"yes: {len(yes_accts):>4} ({yes_pos/max(len(yes_accts),1):.1%} pos) | "
          f"no: {len(no_accts):>4} ({no_pos/max(len(no_accts),1):.1%} pos)")
    asked.add(q.technique)
    game.answer(q.technique, True)

print()
print("=" * 70)
print("FULL-WORKLIST QUESTION VALUE (first question chosen over all 1000,")
print("ranked by information gain -- descriptive positive rates alongside)")
game2 = TwentyQuestions(population)
gains = []
for t, users in game2.by_technique.items():
    yes = users & game2.candidates
    no = game2.candidates - yes
    if not yes or not no:
        continue
    p = len(yes) / len(game2.candidates)
    h_after = p * entropy(len(yes)) + (1 - p) * entropy(len(no))
    gain = entropy(len(game2.candidates)) - h_after
    yes_pos = sum(worklist_labels[a] for a in yes)
    no_pos = sum(worklist_labels[a] for a in no)
    gains.append((gain, t, len(yes), yes_pos / len(yes), len(no), no_pos / len(no)))
gains.sort(reverse=True)
print(f"  {'question':<30}{'gain':>7}{'yes':>6}{'pos%yes':>9}{'no':>6}{'pos%no':>9}")
for gain, t, ny, py, nn, pn in gains[:15]:
    print(f"  {t:<30}{gain:>7.3f}{ny:>6}{py:>9.1%}{nn:>6}{pn:>9.1%}")
never_split = [t for t, users in game2.by_technique.items()
               if not (users & game2.candidates) or not (game2.candidates - users)]
print(f"  facts that never split the worklist (everyone answers alike): "
      f"{sorted(never_split)}")

print()
print("=" * 70)
print("5-SEED CHECK OF THE TRIAGE TIER (seed-0 headline must not be a lucky draw)")
print("cross-currency tier rate within each seed's own top-1000 worklist:")
for seed in [0, 1, 2, 3, 4]:
    tr, te = train_test_split(idxs, test_size=0.4, stratify=label, random_state=seed)
    ytr, yte = label[tr], label[te]
    u = M[tr].T @ ytr
    h = np.asarray(M @ u).flatten()
    Xs = np.column_stack([X_static, h, has_fx])
    sc = StandardScaler()
    mdl = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    mdl.fit(sc.fit_transform(Xs[tr]), ytr)
    s = mdl.predict_proba(sc.transform(Xs[te]))[:, 1]
    top = np.argsort(-s)[:WORKLIST_K]
    top_accts = acct_index[te[top]]
    top_pos = yte[top]
    in_fx = top_accts.isin(fx_accts)
    n_fx = int(in_fx.sum())
    fx_pos = int(top_pos[in_fx].sum())
    rest_pos = int(top_pos[~in_fx].sum())
    n_rest = WORKLIST_K - n_fx
    print(f"  seed {seed}: worklist {int(top_pos.sum())}/1000 pos | "
          f"fx tier {fx_pos}/{n_fx} = {fx_pos/n_fx:.1%} | "
          f"rest {rest_pos}/{n_rest} = {rest_pos/n_rest:.1%}")
