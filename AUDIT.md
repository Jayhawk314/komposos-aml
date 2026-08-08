# AUDIT — independent verification pass (2026-08-07)

An adversarial audit of this repo's claims, performed by an auditor who wrote
none of it, under VERIFICATION.md's protocol. Mandate: try to break the
claims, not improve them. Every number below was printed by an actual run on
this machine during the audit session.

**Verdict: the headline claims survive.** Every rerun reproduced the recorded
numbers digit-for-digit, no label leakage was found, and the model beat the
trivial baselines built to attack it. Four defects were found — all
documentation/protocol-level, none touching a headline number. Each is listed
below with the fix that was applied.

## Environment

Matched VERIFICATION.md's pins exactly: Python 3.10.11, numpy 1.26.4,
pandas 2.3.3, scipy 1.15.3, scikit-learn 1.7.2, pyarrow 22.0.0, Windows 11,
32 GB RAM.

## Reproduction — exact, zero deviations

Six scripts rerun; every printed number matched FINDINGS.md to the last digit:

| script | key numbers, printed = recorded |
|---|---|
| `phase2_robustness.py` | AUROC mean 0.9130, std 0.0042, min 0.9074, max 0.9166; shuffle 0.4987; p@npos 0.3610; seed 0 = 0.9161 / 0.3559; inverted-precision table 4.25% / 13.52% / 38.88% / 36.10%, sel@100 258.8 |
| `phase2_cf_hubexclude.py` | 0.9130 → 0.8987 (−0.0143); p@100 0.0425 → 0.8660; p@500 0.1352 → 0.6940; p@1000 0.3888 → 0.5724; p@npos 0.3610 → 0.4112; lift 3.4x → 70.2x; same 15 bank-70 hubs, popularity gap 93 → 567, 1,590 emptied rows |
| `phase2_invariants.py` | seed-0 with_fxflag AUROC 0.9012; 5-seed means 0.8991 / 0.8740 / 0.7176 / 0.5912 / 0.4207; both dead invariants 0 violations; FX population 3,710 / 598 = 16.1186% (13.06x); 54 high-deviation accounts, 0 positive; max_fx_dev alone 0.6090 |
| `phase3_patterns_audit.py` | 370 attempts, 3,209 pattern transactions, 3,170 accounts (all label-positive, all in graph); background 3,187 / 6,357 = 50.1%; every per-typology recall matched (SCATTER-GATHER 99.3% … BIPARTITE 28.9%); FX tier 270 accounts, 264 positive, 6 false positives, background 0.4% |
| `phase1_baseline.py` | degree 0.7650 / 0.0651 / shuffle 0.5007 (whole population, k=6,357 — see defect 2/3) |
| `phase2_cycles.py` | degree 0.7640 / 0.0662 / 0.5027 and reciprocal_count 0.5390 / 0.0857 / 0.4994 on the seed-0 test split — the leaderboard's actual source |

**Not rerun**: `phase6_transfer_hi_large.py` — VERIFICATION.md requires ~25+ GB
free RAM; the machine had 12.0 GB free of 31.3 GB at check time. Attempting it
risked a thrashing failure that would prove nothing. `phase5` was likewise not
rerun. The HI-Medium/HI-Large numbers rest on the recorded runs plus the static
audit below.

## Static audit — the eight protocol invariants

Checked across all 33 phase scripts:

- **No `try:` anywhere** (invariant 6 holds globally, verified by grep).
- All splits are `train_test_split(..., test_size=0.4, stratify=label,
  random_state=seed)` over `sorted` account keys (invariant 2 — one exception,
  defect 3 below).
- All account keys are `bank:account` via the same string construction
  (invariant 1).
- `precision_at_k` is threshold-inclusive (`scores >= threshold`) in every one
  of its 20+ definitions (invariant 5).
- Distinct-score counts printed everywhere (invariant 4).
- Verified empirically that the two split styles in use (index-split vs
  `train_test_split(X, y, ...)` in `phase1_combined.py` /
  `phase2_combined_all.py`) select **identical test rows**, so "byte-identical
  split" holds across styles.
- `phase6_transfer_hi_large.py`'s bincount reimplementation matches its
  docstring equivalence arguments on reading; the CF vector is still
  train-only; the 800M-wedge tripwire raises rather than catches.

## Leakage audit

**No test-label leakage found.** Specifically:

- The CF propagation vector is `M[train_idx].T @ y_train` in every script that
  builds it (`phase2_robustness.py:140`, `phase2_cf_hubexclude.py:169`,
  `phase2_invariants.py:236`, `phase3_patterns_audit.py:177`,
  `phase6_transfer_hi_large.py:211`). Graph structure is transductive, as
  stated, not hidden.
- `StandardScaler` is fit on train rows only, everywhere.
- The hub cutoff reads only counterparty popularity (no labels); the 94–566
  empty gap is confirmed in run output (max kept 93, min excluded 567).
- The FX flag reads only the two currency columns.

**Judgment on the has_fx post-hoc asterisk** (FINDINGS.md states the feature
was noticed from a label breakdown, then validated): **adequate, with one
caveat.** The zero/nonzero split has no tunable parameter, and the improvement
held on all 5 seeds individually (p@500: 0.6960>0.6680, 0.7240>0.7060,
0.7200>0.6960, 0.7320>0.6980, 0.7160>0.7020; p@1000 likewise all five). The
caveat: those 5 seeds re-slice the same accounts the pattern was noticed on,
so seed-consistency alone is weak protection for a post-hoc feature. What
actually rescues it is the frozen transfer evidence — the FX tier held at
86.5% (LI-Small), 99.58% (HI-Medium), 95.89% (HI-Large) on data the flag was
never noticed on.

## Defects found, ranked — and the fixes applied

1. **The shuffle control is weaker than CLAUDE.md described.** (Moderate —
   discipline gap, not a wrong number.) Every script shuffles *test* labels
   and re-grades the *same, already-computed* score vector; CLAUDE.md said
   "labels shuffled, same method re-scored." The implemented control validates
   the grading arithmetic but **cannot detect test-label leakage**: scores
   that literally memorized the test labels would give real AUROC 1.0 and
   shuffled AUROC ~0.5, passing the control. (No such leakage exists — see
   above — but the control couldn't have caught it.)
   **Fix applied**: CLAUDE.md's description corrected to state what the
   control actually does and does not catch; a matching note added to
   VERIFICATION.md invariant 3.
2. **Leaderboard provenance mislabel.** FINDINGS.md's degree row
   (0.7640 / 0.0662 / 0.5027) cited `phase1_baseline.py`, which actually
   prints the whole-population variant (0.7650 / 0.0651 / 0.5007). The
   recorded numbers are real — they are printed by `phase2_cycles.py` on the
   seed-0 test split (verified by rerun).
   **Fix applied**: leaderboard row re-attributed to `phase2_cycles.py`, with
   a footnote explaining both numbers.
3. **`phase1_baseline.py` does not follow the fixed protocol.** It has no
   train/test split — it scores its label-free features on the whole
   population, k=6,357 — so FINDINGS.md's "identical across every script"
   header overstated. Nothing is inflated (the features read no labels), but
   invariant 2 fails for this one script.
   **Fix applied**: exception noted in FINDINGS.md's protocol header and in
   VERIFICATION.md's known-notes section.
4. **REPORT.md glossary contradicted FINDINGS.md on hub counts.** It said
   "Fifteen of them exist in every IBM dataset tested"; FINDINGS.md records
   15 on HI-Small and LI-Small, 17 on HI-Medium, 149 on HI-Large.
   **Fix applied**: glossary corrected.

**Informational, no fix**: the conversion code that produced
`hi_medium_full.parquet`, `hi_large_slim.parquet`, and `li_small_full.parquet`
is not in the repo; `hi_large_slim.parquet` bakes in precomputed
`from_acct`/`to_acct` columns, so HI-Large's account-key construction is not
auditable from repo code. VERIFICATION.md discloses this. The hub names and
row counts matching across datasets argue the conversion was faithful.

## Controls the authors did not run — added by this audit

Script: `audit_baseline_control.py` (added to the repo by the audit; identical
protocol — sorted keys, 60/40 stratified splits, tie-inclusive p@k, shuffle
control, 5 seeds).

**Control 1 — trivial "rank by has_fx, then degree" baseline.** Could two
sorted columns match the model? No (5-seed means):

```
                     AUROC    p@100    p@500   p@1000   p@npos
fx_then_degree      0.8040   0.2448   0.1213   0.0925   0.1435
fx_then_numtx       0.7477   0.0867   0.0661   0.0662   0.1214
degree_alone        0.7665   0.1592   0.1228   0.1167   0.0732
operational model   0.8991   0.8740   0.7176   0.5912   0.4207
```

The model's top-100 precision is 3.6x the best trivial baseline's, and the gap
widens down the list. Honest side-note: fx_then_degree's 0.8040 AUROC would
rank 3rd on the bulk leaderboard, ahead of every non-CF method — a cheap
two-column ranker is a strong bulk ranker here. The CF machinery earns its
keep at the top of the worklist, which is the repo's own operational standard.

**Control 2 — is the 97.8% FX-tier purity just composition?** FX accounts are
16.12% positive overall; maybe the tier is just "big FX accounts." No: the
top-270 test FX accounts ranked by degree alone are 52/270 = 19.26% positive
(top-100: 28.00%), versus 264/270 = 97.8% for FX accounts inside the model's
top-1000. The purity comes from the model's ranking, not FX membership plus
size.

## What a reader should take from this

The quotable pair — 0.90 AUROC on HI-Small, 0.81 on true LI-Small, 87.4%
precision@100 operationally — reproduced exactly, is leakage-free as far as
static analysis can establish, and survived two adversarial controls the
authors never ran. The defects found were documentation-level and are fixed as
of this file's commit.
