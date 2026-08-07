# CLAUDE.md — read this first, then WHY.md and PLAN.md in full

James directs this work. Explain every term. Short answers. Yes/no where a yes/no exists. Runs are truth — quote exact numbers from the actual output, never restate from memory or round. Bad news first, plainly. Tell the finding before writing files.

## What this repo is

KOMPOSOS-AML: score-first anti-money-laundering detection on IBM's synthetic HI-Small dataset (5,078,345 transactions, 515,088 accounts, 1.2342% flagged at the account level). Successor to KOMPOSOS-SEC (`C:\Users\JAMES\komposos-sec-master`), which spent a year building detection methods before it could score any of them — the first scoring run put the actual attacker's machine 104th of 15,683. This repo inverts the order: score first, build second. Read `WHY.md` for the measured numbers behind that decision and `PLAN.md` for the phased approach.

## The discipline — load-bearing, not style

- Every scored method gets, in the same run: a held-out test split (60/40 stratified, `random_state=0`, built from `sorted(all_accts)` — **identical across every script in this repo** so results are directly comparable), a shuffle control (labels shuffled, same method re-scored — should collapse to ~0.5 AUROC; if it doesn't, the harness is lying), a distinct-score-value count (catches universal-tie bugs — a tie can read as a perfect score), and precision@k with ties handled by threshold-inclusive selection, never "count strictly higher."
- **Before any result becomes a headline, run `phase2_robustness.py` on it**: multiple seeds (a single split can be a lucky draw with only 6,357 positives) and precision at operational k (100/500/1000). Both checks caught things the standard single-seed, k=n_positives protocol missed entirely.
- No `try/except` around a scoring path. If something breaks, it raises — three defects in the previous repo failed silently instead.
- No time-window features without a stated, non-label-tuned justification. Picking a window until a signal appears is a threshold in disguise; it killed two detectors in the previous repo.
- Before proposing a method, name the column it reads. If that column doesn't exist in the data, the method cannot work, however good it is.
- A win that only holds on real data and not under a shuffle control isn't a win. A win that's actually a confound one level up (e.g. hub-account bias, ring-membership overlap) needs a second, more targeted control before it's trusted — see `MATH_IDEAS.md`'s closing lesson.

## Where things are

- `WHY.md` / `PLAN.md` — why this repo starts the way it does, and the phased plan. Read in full before adding a method.
- `FINDINGS.md` — the running leaderboard and caveats. **Update every time a new method is scored.**
- `MATH_IDEAS.md` — catalog of ideas surveyed from the old repo's 109 math-adjacent files, split into tried / promising-untried / no-honest-mapping. Consult before re-surveying those folders — already done once, thoroughly, file by file.
- `phase0_inspect.py` — confirms the data has a usable label before anything else. (Rocketgraph's hosted AML dataset does NOT — stripped the label. Don't re-download it.)
- `phase1_*.py` — counting-only baseline. The number every later method has to beat.
- `phase2_*.py` — structural/graph/spectral/collaborative-filtering methods, each scored against the identical split.
- `twenty_questions.py` — ported unmodified from KOMPOSOS-SEC. Self-contained (`TwentyQuestions` class, `play()`, `evaluate()`), takes any `Dict[str, Set[str]]` candidate→features population. Applied to the operational worklist in `phase2_twentyq_worklist.py` (2026-08-07): identification capped by signature collisions (441 distinct signatures / 1000 suspects), but the audit surfaced a near-pure triage tier (cross-currency ∩ worklist = 97.8% positive) — see FINDINGS.md's twenty-questions section.

## Known defect — FIXED (2026-08-07, `phase2_cf_hubexclude.py`)

The best method had a measured operational defect: precision@100 4.25% vs precision@1000 38.88% — precision *increased* going down the ranking, because accounts whose only counterparty was the giant hub `70:100428660` (14,775 touchers) tied at the top as false positives. **Fixed by explicit hub exclusion** (drop the 15 counterparties with popularity > 100 — a cutoff sitting in a literal empty gap in the distribution, not label-tuned): p@100 4.25% → 86.60%, p@500 13.52% → 69.40%, p@1000 38.88% → 57.24% (5-seed means), at a stated cost of −0.0143 AUROC (0.9130 → 0.8987). See FINDINGS.md's hub-fix section. Steeper *down-weighting* had been tried first and failed for a structural reason (row-normalization erases IDF for single-counterparty accounts).

Lesson to carry: **always report precision at operational k (100, 500, 1000), not just k=n_positives.** A single aggregate metric hid a defect that made the top of the worklist nearly useless.

## Explainability

`phase2_explain.py` traces any account's `cf_damped` score back to specific shared counterparties and specific known-bad training accounts — not a black box. Found a real, still-open weakness this way: false positives cluster around one giant-hub counterparty (14,775 touchers) that isn't damped enough. Two attempted fixes (amount-weighting, steeper IDF) both made overall accuracy worse — see FINDINGS.md's "Explainability" section before trying a third.

## Current best (full table in FINDINGS.md)

Operational: `combined_with_cf` with hub exclusion + binary cross-currency flag (`phase2_invariants.py`, "with_fxflag"): **0.8991 AUROC, 87.40% p@100, 71.76% p@500, 59.12% p@1000** (5-seed means). Bulk-AUROC best remains the unexcluded `combined_with_cf` (`phase2_cf_combined.py`): **0.9161 AUROC, 0.3559 precision@k**.

**Standing rule (2026-08-07, `phase3_transfer_li_full.py`): never quote the HI-Small numbers without the transfer numbers — the quotable pair is 0.90 AUROC on HI-Small, 0.81 on true LI-Small** (62.6x lift@100, 86.5% fx-tier, frozen recipe, zero re-tuning, different generation run). The earlier 0.69 on the HF mirror was ~half subsample starvation, and most of the remaining 0.09 gap is label composition — 78.0% of LI's positives are structureless background vs 50.1% on HI, while the per-typology skill ordering transfers exactly (`phase4_li_typology_audit.py`). Decomposition in FINDINGS.md's Phase 3 section. `data/archive.zip` holds all six IBM variants plus `*_Patterns.txt` typology ground-truth files (HI-Small_Patterns.txt extracted — never yet used; it names which transactions form which laundering pattern). Damped collaborative filtering over shared counterparties (ported from `cyber/cooccurrence_predictor.py` — the one method that was already the best-measured performer in KOMPOSOS-SEC's own ATT&CK benchmark, 0.4375 MRR there, beating every categorical/topological method tried on that task) combined with counting/cycle features.

**Resolved** (`phase2_cf_verify.py`): checked whether this is mostly "shares a component with a training-set positive" (a cruder finding). It isn't — 72.24% of the whole network sits in a component with a training positive, so reachability alone only lifts the local base rate 1.38x. CF's precision *within* that reachable population is still 34.05% (19.92x local lift, AUROC 0.8652 there vs. shuffle control 0.4990) — real discriminative work, not ring-adjacency riding along. 0.9161 stands as the current best.

## Conventions for new phase scripts

- Load both parquet files (`data/hi_small_0.parquet`, `data/hi_small_1.parquet`), concatenate.
- Account identity is `(Bank, Account)`, **not** `Account` alone — 4 accounts in file 0 collide across banks under `Account` alone.
- Build the same 60/40 stratified split, seed 0, from the same account ordering, so results compare directly against every other script without re-deriving anything.
- Print the full score table (AUROC, AUROC on shuffled labels, distinct score count, precision@k, k, hits, selected) for every feature — including losing ones. A losing result is still a result.
- No unprompted method-building — check in on direction before starting something with real scope (a new data structure, a new external library, anything PLAN.md calls "the real work").
- Update `FINDINGS.md` with the result before moving on to the next thing.

## What NOT to do

- Don't reach for `categorical/`, `zfc/`, `topology/`, `hott/`, `cubical/` wholesale — those exact modules measured losing to plain counting on KOMPOSOS-SEC's one scored task (WHY.md section 3), and porting a 500–1000 line ATT&CK-specific file onto a transaction graph risks landing exactly there again, or worse, silently broken (the ZFC verification adapter's documented history: never executed, returned empty agreement that read as success). `MATH_IDEAS.md` already extracted the genuinely transferable ideas from those files — start there, build small and honest, don't port wholesale.
- Don't pick a time window to find a threshold that makes a number look good.
- Don't trust a score without its shuffle control, distinct-score count, and precision@k population size checked.
