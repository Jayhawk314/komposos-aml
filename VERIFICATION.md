# VERIFICATION — for an independent auditor

This file exists so that someone who did not write this repo can check it.
Runs are truth: every claim below names the script that prints it.

## Environment (the one every recorded number came from)

- Python 3.10.11 on Windows 11, 32 GB RAM
- numpy 1.26.4, pandas 2.3.3, scipy 1.15.3, scikit-learn 1.7.2, pyarrow 22.0.0

All scripts are seeded; on this environment, reruns reproduce the recorded
numbers exactly. Different library versions may shift values in the last
decimal place (train_test_split and LogisticRegression internals); protocol
conclusions should not change.

## Data required (not in the repo — see NOTICE)

| file in `data/` | source | used by |
|---|---|---|
| `hi_small_0.parquet`, `hi_small_1.parquet` | Hugging Face mirror (URLs in README) | all phase0–phase4 scripts |
| `HI-Small_Patterns.txt` | Kaggle archive | `phase3_patterns_audit.py`, `phase4_*` |
| `li_small_full.parquet` | `LI-Small_Trans.csv` from Kaggle archive, converted (`pd.read_csv(...).to_parquet(...)`) | `phase3_transfer_li_full.py`, `phase4_li_typology_audit.py` |
| `LI-Small_Patterns.txt` | Kaggle archive | `phase4_li_typology_audit.py` |
| `hi_medium_full.parquet` | `HI-Medium_Trans.csv`, chunk-converted | `phase5_transfer_hi_medium.py` |
| `hi_large_slim.parquet` | `HI-Large_Trans.csv`, chunk-converted with precomputed `from_acct`/`to_acct` and slim columns | `phase6_transfer_hi_large.py` |
| `li_smaller_train/test.parquet` | HF `qubit420/ibm-aml-LI-smaller` (a 50% subsample — superseded by true LI-Small) | `phase3_transfer_li.py` only |

## Protocol invariants — check these in every scoring script

1. Account identity is `(Bank, Account)` joined as `f"{bank}:{account}"` —
   never `Account` alone.
2. The split: `train_test_split` with `test_size=0.4`, `stratify=label`,
   `random_state=seed`, over indices of `sorted(all_accts)` — byte-identical
   across scripts so results compare directly.
3. A shuffled-label AUROC is computed in the same run and should be ~0.50.
4. Distinct-score count is printed (guards against universal-tie artifacts).
5. `precision_at_k` uses threshold-inclusive selection (ties at the cutoff
   included; `sel` may exceed `k`) — never "count strictly higher."
6. No `try/except` around any scoring path.
7. The CF propagation vector is built from TRAIN labels only
   (`M[train_idx].T @ y_train`); graph structure may use all rows
   (transductive — stated, not hidden).
8. The hub cutoff (popularity > 100) and the FX flag are label-free
   constructions; the cutoff sits in a measured empty gap (94–566) of the
   counterparty-popularity distribution.

## Headline claims → the script that prints them

| claim | script | expected (seed 0 / 5-seed mean) |
|---|---|---|
| Operational best: 0.8991 AUROC, p@100 0.8740, p@500 0.7176, p@1000 0.5912 (5-seed means) | `phase2_invariants.py` (with_fxflag rows) | seed 0 AUROC 0.9012 |
| Hub fix: p@100 4.25% → 86.60% at −0.0143 AUROC | `phase2_cf_hubexclude.py` | verdict table at end |
| CF not mere ring-adjacency | `phase2_cf_verify.py` | 34.05% precision within reachable population |
| Robustness of pre-fix model + the tie-block defect | `phase2_robustness.py` | p@100 4.25%, sel@100 258.8 |
| Transfer, true LI-Small: 0.8075 AUROC | `phase3_transfer_li_full.py` | ladder table |
| Transfer, HI-Medium: 0.8736 AUROC, fx tier 99.58% | `phase5_transfer_hi_medium.py` | note: run recorded in FINDINGS printed "LI" in two headers (template artifact, since fixed) |
| Transfer, HI-Large: 0.8450 AUROC on 179.7M rows | `phase6_transfer_hi_large.py` | ~30–45 min, needs ~25+ GB free RAM |
| 50.1% of HI-Small positives are background; STACK/BIPARTITE blind spots | `phase3_patterns_audit.py` | recall table |
| 78.0% of LI positives are background; skill ordering transfers | `phase4_li_typology_audit.py` | recall table |
| Butterfly + flow-ratio blind-spot attempts both negative | `phase4_butterfly.py`, `phase4_flowratio.py` | verdict tables |
| FX tier 97.8% pure, stable across 5 seeds | `phase2_twentyq_worklist.py` | final section |

Negative results (2-hop CF, amount-weighted CF, strong IDF, Dempster-Shafer,
coherence, PageRank, Kan extensions, typologies) each have their own script;
FINDINGS.md's leaderboard maps them.

## Known implementation notes an auditor should not mistake for bugs

- `phase6_transfer_hi_large.py` reimplements the same feature definitions
  with pyarrow + np.bincount; equivalence arguments are in its docstring.
  Its triangle build has an 800M-wedge feasibility tripwire that raises
  (measured infeasibility) rather than thrashing swap.
- The FX flag has two near-identical definitions: "nonzero rate deviation"
  (`phase2_invariants.py`) vs "any cross-currency tx" (later scripts); they
  differ only for accounts whose implied rate exactly equals the pair
  median (AUROC 0.9012 vs 0.9010, seed 0). Documented in FINDINGS.md.
- `phase2_invariants.py` prints two dead invariants (0 violations) on
  purpose — honesty about rules the generator enforces.
- The has_fx feature's provenance is post-hoc (noticed from a label
  breakdown, then validated across 5 seeds) — FINDINGS.md states this
  asterisk explicitly.

## What a finding of "does not reproduce" should look like

Name the script, the environment versions, the printed number, and the
recorded number side by side. FINDINGS.md is falsified by runs, not by
argument.
