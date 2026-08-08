# FINDINGS — running results log

Every number below is from an actual run in this session (2026-08-07), reproducible via the named script. Update this file every time a new method is scored — don't let a result exist only in conversation.

**Fixed protocol, identical across every script** (one exception: `phase1_baseline.py` predates the protocol and scores its label-free features on the whole population, k=6,357 — AUDIT.md defect 3): held-out test split 60/40 stratified, `random_state=0`, built from `sorted(all_accts)` so the split is byte-identical everywhere. 206,036 test accounts, 2,543 positive, base rate 1.2342%. Every method reports: AUROC, AUROC on shuffled labels (should collapse to ~0.5 — a control, not decoration), distinct score count, precision@k with ties handled by threshold-inclusive selection.

## Leaderboard (bulk ranking: AUROC / precision@k, k = 2,543)

| rank | method | script | AUROC | prec@k | shuffle AUROC |
|---|---|---|---|---|---|
| 1 | **combined_with_cf** (logreg: counting + cycles + cf_damped) | `phase2_cf_combined.py` | **0.9161** | **0.3559** | 0.5027 |
| — | **combined_with_cf, hub-excluded, + has_fx flag** — **operational recommendation**: adds the binary cross-currency flag; p@500/p@1000 up on all 5 seeds, AUROC flat — see invariants section | `phase2_invariants.py` | 0.9012 | **0.4168** | 0.5041 |
| — | combined_with_cf, hub-excluded (previous operational rec.): p@100 0.89 vs 0.03 (seed 0); AUROC slightly lower than unexcluded — see hub-fix section | `phase2_cf_hubexclude.py` | 0.9036 | 0.4082 | 0.5032 |
| 2 | cf_damped alone (damped collaborative filtering) | `phase2_cf.py` | 0.8832 | 0.3405 | 0.5010 |
| 3 | combined_all (logreg, 10 counting/cycle features) | `phase2_combined_all.py` | 0.7759 | 0.0861 | 0.5060 |
| 4 | combined (logreg, 6 counting features) | `phase1_combined.py` | 0.7762 | 0.0653 | 0.5023 |
| 5 | degree (true counterparty union) | `phase2_cycles.py`† | 0.7640 | 0.0662 | 0.5027 |
| 6 | in_degree | `phase2_fan.py` | 0.7391 | 0.0737 | 0.5019 |
| 7 | out_degree | `phase2_fan.py` | 0.7002 | 0.0299 | 0.5067 |
| 8 | typology_best (Jaccard match, 8-typology catalog) | `phase2_typologies.py` | 0.5871 | 0.0309 | 0.5016 |
| 9 | in_nontrivial_scc (any-length cycle) | `phase2_scc.py` | 0.6016 | 0.0673 | 0.5046 |
| 10 | pagerank_amount | `phase2_spectral.py` | 0.5665 | 0.0495 | 0.5032 |
| 11 | pagerank_count | `phase2_spectral.py` | 0.5437 | 0.0436 | 0.5032 |
| 12 | left_kan (Kan extension, max-propagation) | `phase2_kan.py` | 0.5466 | 0.0624 | 0.4997 |
| 13 | reciprocal_count (2-cycle) | `phase2_cycles.py` | 0.5390 | 0.0857 | 0.4994 |
| 14 | tri3_count (3-cycle) | `phase2_cycles3.py` | 0.5076 | n/a* | 0.4998 |
| 15 | right_kan (Kan extension, min-propagation) | `phase2_kan.py` | 0.4068 | 0.0072 | 0.5004 |
| — | combined_1hop_2hop (adding 2-hop CF to the hub-excluded combined model — worse, see caveat) | `phase2_cf_2hop.py` | 0.8956 | 0.4066 | 0.5012 |
| — | cf_2hop alone (hub-excluded matrix) | `phase2_cf_2hop.py` | 0.8865 | 0.3299 | 0.5025 |
| — | dempster_shafer (degree + cf_damped, conflict-aware combination) | `phase2_dempster_shafer.py` | 0.8985 | 0.2497 | 0.5021 |
| — | cf_amount_weighted (CF weighted by $ instead of binary touch) | `phase2_cf_amount.py` | 0.8825 | 0.3052 | 0.5008 |
| — | best_coherence (amount-conservation around cycles) | `phase2_coherence.py` | 0.5416 | 0.0901 | 0.4994 |
| — | random_noise | `phase1_baseline.py` | 0.5004 | 0.0131 | 0.4973 |

*tri3_count's precision@k is degenerate at k=2,543 (only 316 test accounts have any value at all — see "honest sparse-feature question" below).

†Provenance corrected by the 2026-08-07 audit (AUDIT.md defect 2): this row was previously attributed to `phase1_baseline.py`, which actually prints the whole-population variant (0.7650 AUROC / 0.0651 prec@k, k=6,357, no split). The row's numbers are printed by `phase2_cycles.py` on the standard seed-0 test split.

## Best narrow-population (high-lift, small-net) findings

Bulk AUROC undersells these — they catch few accounts but very concentrated ones:

| finding | population | positive | rate | lift |
|---|---|---|---|---|
| has any cross-currency transaction (test set) | 3,710 | 598 | 16.12% | **13.06x** |
| cycle3_and_bank (3-cycle + multi-bank, whole dataset) | 748 | 110 | 14.71% | 11.92x |
| SCC excluding giant component (<100 members, test set) | 2,453 | 238 | 9.70% | 7.86x |
| round_robin_layering typology group (whole dataset) | 1,016 | 90 | 8.86% | 7.18x |
| reciprocal_count>0 (2-cycle, test set) | 2,671 | 229 | 8.58% | 6.95x |
| left_kan "new" (paid by a round-robin account, not one itself) | 4,860 | 131 | 2.70% | 2.18x |

**Honest sparse-feature question**: when a feature is nonzero for far fewer accounts than k, precision@k is degenerate (threshold falls to 0, "top k" silently includes the whole population, number equals base rate exactly). Always check population size before trusting precision@k on a sparse feature — report the rate *within the nonzero population* instead.

## Caveats that matter — do not treat these as resolved

- **cf_damped / combined_with_cf is transductive**: uses full graph structure plus TRAIN labels only (no test-label leakage, verified in the code path). **RESOLVED** (`phase2_cf_verify.py`): checked whether the lift is mostly "shares a component with a training-set positive" (a cruder finding) — it isn't. 72.24% of the whole network (372,110 of 515,088 accounts) sits in a component with a training positive, so mere reachability only lifts the local base rate 1.38x (1.71% vs. 1.23%). CF's precision *within* that reachable population is still 34.05% — a 19.92x LOCAL lift, AUROC 0.8652 there vs. shuffle control 0.4990. CF is doing real discriminative work inside a population where cluster membership alone is nearly uninformative, not just detecting ring adjacency. Side-finding: all 57,256 test accounts unreachable from any training positive were negative (0 positives).
- **degree vs degree_approx**: the true counterparty union (dedup across in/out direction) barely moved AUROC vs. the naive double-counted sum (0.7650 vs 0.7662) but did change precision@k (0.0651 vs 0.0722, more ties at the cutoff).
- **PageRank and the literal Kan extension both lost to plain counting** — the third and fourth independent instance (after two in KOMPOSOS-SEC, WHY.md section 3) of elaborate/spectral math losing to arithmetic on this project's tasks. Collaborative filtering (also "elaborate" by the same survey) reversed that pattern decisively — see MATH_IDEAS.md for why CF is different (measured weights, not guessed ones; explicit anti-hub damping).
- **right_kan's vacuous-limit confound**: ~209k no-outgoing-edge accounts all default to the "top" element under the correct category-theoretic convention, which happens to be the wrong default for this population (they're under-represented among launderers) — a real math construction producing a real artifact, not a bug. Full explanation in the phase2_kan.py run.
- **Dempster-Shafer conflict mass is a real negative filter, though the combined score itself doesn't lead**: combining degree + cf_damped via Dempster-Shafer (`phase2_dempster_shafer.py`) scored 0.8985 AUROC / 0.2497 prec@k — beats cf_damped alone on AUROC but loses to it (and to the logreg combination) on precision@k. The genuinely useful output is the conflict-mass diagnostic: accounts where degree and cf_damped actively disagree (top 10% conflict) run 0.0383% positive vs. 1.8925% where they roughly agree (bottom 90%) — a 49x gap. Disagreement between the two strongest detectors is a strong signal of "nothing here," not ambiguity — useful as a deprioritization filter, not as a ranking feature on its own.
- **Amount-weighting CF does not help**: tested (`phase2_cf_amount.py`) whether weighting the collaborative-filtering matrix by dollar amount instead of binary touch improves on cf_damped. It doesn't — AUROC is a wash (0.8825 vs 0.8832) and precision@k is worse (0.3052 vs 0.3405, 90 fewer hits). Third data point (after PageRank's count-vs-amount comparison) suggesting this dataset's "the graph has weights" property matters much less than WHY.md's live hypothesis expected — binary structure keeps beating weighted structure.
- **Structural invariants are mostly generator-enforced, and the FX-deviation hypothesis came back INVERTED** (`phase2_invariants.py`): same-currency Paid≠Received fires 0 times in 5,006,175 rows; Reinvestment-not-self-loop fires 0 times in 481,056 — both dead by construction. The one live rule, FX-rate consistency, discriminates *backwards*: 0 of the 54 test accounts with ≥10% rate deviation are positive, while positives concentrate in the ≤0.1%-deviation band (17.07%). Launderers in this generator use exact market rates; the deviants are innocent (Bitcoin's genuine 2x drift). max_fx_dev alone: 0.6090 AUROC, no marginal value in the combined model. **The salvage — the has_fx binary flag — is real**: cross-currency accounts run 16.12% positive (13.06x lift), and adding the flag (natural zero/nonzero split, no tuned threshold) to the operational best improves p@500 0.6940 → 0.7176 and p@1000 0.5724 → 0.5912, both up on all 5 seeds individually, AUROC flat (+0.0004). Provenance stated honestly: this feature was noticed from the label breakdown of the FX population, not hypothesized in advance — trusted because of 5-seed consistency and the untuned split, but it carries that post-hoc asterisk.
- **Both targeted shots at the STACK/BIPARTITE blind spots failed, and the failure is structural** (`phase4_butterfly.py`, `phase4_flowratio.py`). Butterfly counting (pairs of accounts paying the same pair of counterparties — the classic bipartite-density signature, hub-guarded): STACK recall@npos 39.0% → 39.4%, BIPARTITE 28.9% → 28.9%; only 12% of those typologies' accounts have any butterfly at all — reading the attempts directly shows why: the generator's BIPARTITE is disjoint single edges (13 tx, 26 distinct accounts, no account repeats) and STACK is disjoint 2-hop chains, all with fresh counterparties. Flow ratio (min/max of total in/out amounts, targeting STACK pass-through middles): STACK 39.0% → 39.0%, and AUROC degrades 0.8958 → 0.8741 — the single pass-through event drowns in aggregate (STACK median flow_ratio 0.3631, lower than most typologies). Fourth amount-based negative. **Conclusion: these two typologies leave no account-level aggregate trace by construction; their 29–39% recall rides on the accounts' other activity and is a ceiling for this entire class of methods.** The only remaining doors are transaction-level scoring or temporal receive→forward chaining — the latter requires confronting the repo's time-window rule head-on as a named methodological decision, not a feature. (Honest side-note: butterflies gave +0.0048 AUROC overall with targets unmoved — not adopted; the criterion was target recall, and features without a purpose are how leaderboards rot.)
- **2-hop CF does not help** (`phase2_cf_2hop.py`, hub-excluded matrix, S² y computed as matvecs so no memory blowup): adding 2-hop to the combined model is worse on 5-seed means — AUROC 0.8987 → 0.8896, p@100 0.8660 → 0.8360, p@500 0.6940 → 0.6692, p@1000 flat, p@npos +0.0064. Nuance worth keeping: cf_2hop *alone* beats cf_1hop alone on AUROC (0.8865 vs 0.8600, seed 0) via much broader coverage (109,356 distinct scores vs 30,239 — it reaches accounts 1-hop can't), but the added signal is diffuse and lower-precision (prec@k 0.3299 vs 0.4019), and in combination it subtracts. **CF recipe is now a confirmed local optimum**: amount-weighting, steeper IDF, and 2-hop all negative; hub exclusion is the only modification that ever helped, and it helped operationally, not on AUROC. Stop extending CF.
- **Amount-coherence around cycles does not work on this dataset**: tested (`phase2_coherence.py`) whether real round-robin cycles conserve dollar value hop-to-hop vs. incidental cycles among unrelated amounts. Within the cycle population, high-coherence (≥0.9) accounts were 9.88% positive vs. 8.92% for low-coherence — statistically indistinguishable (n=243 vs n=2,443). The bulk-ranker score is identical to plain "has any cycle" — the continuous coherence value adds nothing once cycle membership is already known. Either the synthetic generator doesn't encode amount-conservation in its laundering typologies, or fee/randomization noise washes it out.

## Twenty questions on the operational worklist (phase2_twentyq_worklist.py)

`twenty_questions.py` applied UNMODIFIED to a real candidate set for the first time: the seed-0 top-1000 worklist (578 positive = 57.80%), features = 31 natural zero/nonzero binary facts (no thresholds, none label-built).

- **Identification ceiling is low**: 441 distinct signatures among 1,000 suspects — most of the worklist is structurally identical under binary facts. Info strategy: mean 9.0 questions (floor log2(1000) = 10.0), identifies 27% within 20; popularity baseline: 19.0 / 18%. Largest identical block: 64 accounts (44 positive), signature `US Dollar + ACH + Reinvestment + multi_bank + receives + self_loop + sends`. `multi_bank` never splits — the entire top-1000 is multi-bank.
- **Descriptive triage splits inside the worklist** (reported rates, not a scored detector): cross_currency 270 accts **97.8% positive** vs 43.0% without; no-ACH 152 accts 3.3% positive; receives-nothing 114 accts 5.3%; no-self-loop 176 accts 22.2%. Operational reading: the 270 cross-currency worklist accounts are a near-pure first tier; three "no" answers are strong innocence markers for deprioritization (cf. the Dempster-Shafer conflict-mass finding — this is a second, cheaper deprioritization signal).
- **The tier is stable across all 5 seeds** (same script, per-seed model + worklist rebuilt): fx-tier rate 97.8% / 97.3% / 98.2% / 97.1% / 96.7% (n = 270/292/274/276/272), remainder 43.0–45.6%. Not a seed-0 draw.
- Minor: the rebuilt model printed AUROC 0.9010 vs phase2_invariants.py's 0.9012 — this script defines the flag as "any cross-currency tx," invariants used "nonzero rate deviation"; they differ on accounts whose implied rate exactly equals the pair median. Immaterial, noted for reproducibility.

## Phase 3 — frozen transfer test on LI data (phase3_transfer_li.py) — HEADLINE NUMBERS DO NOT TRANSFER

PLAN.md's named risk ("a method that works here may be learning the generator") tested for real: the operational-best recipe, frozen with zero re-tuning, run on `qubit420/ibm-aml-LI-smaller` (HF mirror, no auth; 3,462,024 rows, 595,898 accounts, 2,790 positive = 0.4682%; account overlap with HI-Small: 3 — a genuinely different generation run). **Provenance caveat on every number: true LI-Small is ~6.9M rows, this mirror is ~half — likely a subsample, so the graph is thinner than IBM shipped.**

| metric (5-seed means) | HI-Small | LI mirror |
|---|---|---|
| AUROC | 0.8991 | **0.6901** |
| shuffle AUROC | 0.4975 | 0.5024 |
| p@100 | 0.8740 | 0.2580 |
| p@500 | 0.7176 | 0.2332 |
| p@1000 | 0.5912 | 0.2030 |
| p@npos | 0.4207 | 0.1980 |
| fx-tier rate in top-1000 | 97.4% | 60.9% |

Honest decomposition:
- **Mechanical part**: LI's base rate is 2.6x lower, so precision shrinks by construction. Base-rate-free, top-100 lift is **55.1x on LI vs 70.8x on HI** — still far above chance, shuffle controls clean.
- **Real part**: AUROC is base-rate-independent and fell 0.21. The graph/CF signal degraded substantially on the new data.
- **What transferred**: the frozen hub rule (popularity > 100) excluded exactly 15 counterparties again — the same bank-70 hub family (`70:10042B660`, 18,171 touchers), so that structure is cross-dataset real. The FX tier still triples the within-worklist hit rate (60.9% vs 18–19.5% remainder) though its near-purity is gone.
- **Confounded attribution**: can't distinguish "learned HI's generator" from "CF starved by the subsampled graph" on this mirror. Disambiguation needs true LI-Small (Kaggle, manual login download).

Standing rule from this result: **quote the recipe as 0.90 AUROC on HI-Small and 0.69 on the LI mirror — never the first without the second.** *(Superseded same day by the true-LI-Small run below — the quotable pair is now 0.90 / 0.81.)*

### RESOLVED same day — true LI-Small run (phase3_transfer_li_full.py) decomposes the drop

James downloaded IBM's full Kaggle archive (`data/archive.zip`, all six variants + Patterns.txt typology ground-truth files). True LI-Small: 6,924,049 rows (the HF mirror was exactly a 50% subsample), 705,907 accounts, 5,304 positive = 0.7514%. Frozen recipe, zero re-tuning, 5 seeds:

| metric (5-seed means) | HI-Small | LI mirror (half graph) | LI true (full graph) |
|---|---|---|---|
| AUROC | 0.8991 | 0.6901 | **0.8075** |
| shuffle | 0.4975 | 0.5024 | 0.4982 |
| p@100 | 0.8740 | 0.2580 | 0.4700 |
| p@500 | 0.7176 | 0.2332 | 0.3352 |
| p@1000 | 0.5912 | 0.2030 | 0.3372 |
| p@npos | 0.4207 | 0.1980 | 0.2797 |
| lift@100 | 70.8x | 55.1x | **62.6x** |
| fx-tier rate | 97.4% | 60.9% | **86.5%** (range 81.2–91.0 across seeds) |
| base rate | 1.2342% | 0.4682% | 0.7514% |

**Decomposition of the mirror's 0.21 AUROC drop: ~0.12 was subsample starvation (half the graph), ~0.09 is the genuine cross-dataset gap.** The frozen hub rule (>100) excluded exactly 15 counterparties on every dataset tested — the same bank-70 hub family each time (`70:10042B660`, 19,727 touchers on true LI) — cross-dataset structural fact, not an HI quirk. Precision differences are partly mechanical (LI base rate is 0.61x HI's).

**Final transfer verdict: the recipe genuinely generalizes — 0.8075 AUROC / 62.6x lift@100 / 86.5% first tier on a different generation run, unretuned.** The 0.09 AUROC gap is real and stays quoted alongside the headline: **0.90 on HI-Small, 0.81 on true LI-Small.**

## Phase 5 — HI-Medium, the scale rung (phase5_transfer_hi_medium.py)

Frozen recipe on HI-Medium from the Kaggle archive: **31,898,238 rows, 2,077,023 accounts, 41,857 positive = 2.0152%**. Ran to completion on the 32 GB laptop — the "memory feasibility unknown" question is answered.

5-seed means: **AUROC 0.8736** (shuffle 0.4992), p@100 0.7900, **p@500 0.8176, p@1000 0.7682** (both *above* HI-Small's 0.7176/0.5912 — partly mechanical, base rate 2.0152% vs 1.2342%), p@npos 0.4077 (npos = 16,743), lift@100 39.2x. **FX tier: 99.58% mean — seed 1 was literally 435/435.** Distinct scores ~787K.

The frozen hub rule (>100) excluded 17 counterparties, topped by **the same account ID as HI-Small's giant hub — `70:100428660`, here with 58,260 touchers**. Third dataset (HI-Small, LI-Small, HI-Medium) where the same bank-70 hub family emerges and the same untuned exclusion rule fires correctly.

Note for reruns: the run whose numbers are quoted above printed "LI" in some section headers — a template artifact from the copied script, since fixed; the loaded file was `hi_medium_full.parquet` (row/account counts confirm).

## Phase 6 — HI-Large: the ladder complete (phase6_transfer_hi_large.py)

Frozen recipe on HI-Large: **179,702,229 rows, 2,116,168 accounts, 222,522 positive = 10.5153%** (account base rate is 8.5x HI-Small's because accounts average ~85 transactions here vs ~10 — more chances to be touched; tx-level rate is a familiar 0.1255%). **Ran to completion on the 32 GB laptop** — but only after rewriting the implementation: pyarrow encoding + np.bincount aggregations over integer account codes (the math is unchanged and equivalence is documented in the script docstring; the pandas string-groupby implementation would not have fit). Wedge count for the triangle build: 217,367,494 — under the 800M feasibility tripwire.

5-seed means: **AUROC 0.8450** (range 0.8444–0.8456 — at 2.1M accounts, seeds barely matter; shuffle 0.5002), p@100 0.8660, p@500 0.7636, p@1000 0.7076, p@npos 0.4414 (npos = 89,009), lift@100 8.2x at the 10.52% base. FX tier 95.89% mean.

Frozen hub rule excludes **149** counterparties at this scale; the top five are byte-identical to HI-Medium's (`70:100428660` 58,260; `70:1004286A8` 36,573; …) — HI-Medium appears to be a subset of the same simulation run (observation, not verified).

**The final ladder (frozen recipe, 5-seed means):** HI-Small 0.8991 → HI-Medium 0.8736 → HI-Large 0.8450 → LI-Small 0.8075 AUROC. The AUROC glide with scale is real and unexplained beyond densification; the operational numbers stay strong everywhere (p@100 ≥ 79% on every HI rung), and the FX tier never drops below 86% on any dataset.

## Typology ground-truth audit (phase3_patterns_audit.py) — reframes the whole leaderboard

`HI-Small_Patterns.txt` (from the Kaggle archive) names all 370 generated laundering attempts: typology + exact transactions. First run against it, seed-0 operational best:

- **50.1% of label-positive accounts (3,187 of 6,357) are in NO pattern** — background laundering the generator emits outside its structured typologies. The model's recall on them: 4.1% in top-1000, 18.1% at k=npos — near-blind, as any structural method must be. **Consequence: the ~42% p@npos ceiling was never a defect — a perfect structural detector tops out near ~50% on this dataset.** All 3,170 pattern accounts are label-positive and present in the graph (parse verified).
- **Per-typology recall at k=npos (top-1000)**: SCATTER-GATHER 99.3% (91.2%), GATHER-SCATTER 93.1% (63.6%), FAN-IN 79.8% (50.8%), FAN-OUT 77.4% (48.4%), CYCLE 71.7% (38.7%), RANDOM 63.9% (33.7%), **STACK 39.0% (16.7%), BIPARTITE 28.9% (12.2%) — the two named blind spots.** Test-set account counts per typology: 197 BIPARTITE, 106 CYCLE, 124 FAN-IN, 159 FAN-OUT, 275 GATHER-SCATTER, 83 RANDOM, 137 SCATTER-GATHER, 269 STACK.
- **The FX tier is not a single-typology artifact**: its 264 positives span all 8 typologies (GATHER-SCATTER 29.5%, SCATTER-GATHER 27.7%, FAN-IN 13.3%, CYCLE 10.6%, STACK 10.2%, rest <6% each, background 0.4%), 6 false positives total.
- Parsing note: bank fields in Patterns.txt are zero-padded (`021174`); normalized via `int()` to match the repo's `bank:account` key. 3,209 pattern transactions parsed; typology extraction regex must not require a colon (BIPARTITE/SCATTER-GATHER/STACK headers have none).

Named future work from this audit: targeted features for STACK (layered chains) and BIPARTITE (dense many-to-many between two account groups) — the only two typologies where most accounts escape even the full top-npos list. *(Tried same day, twice, closed — see the phase4 caveat: account-level-invisible by construction.)*

### LI-Small typology audit (phase4_li_typology_audit.py) — the transfer gap explained

Same audit, frozen model, true LI-Small (101 attempts, 1,168 pattern accounts, all label-positive, all in graph):

- **78.0% of LI's positive accounts are background** (4,136 of 5,304) vs HI's 50.1%. "LI = lower illicit" in practice means a much lower structured-to-background ratio. Since background is near-undetectable by structure everywhere (LI background recall@npos: 21.4%), **most of the 0.09 AUROC transfer gap is label composition, not lost skill.**
- **The typology skill ordering transfers exactly**: recall@npos LI vs HI — SCATTER-GATHER 82.9%/99.3% (best on both), FAN-OUT 79.7%/77.4%, GATHER-SCATTER 74.6%/93.1%, FAN-IN 57.5%/79.8%, CYCLE 53.6%/71.7%, RANDOM 36.4%/63.9%, STACK 30.2%/39.0%, BIPARTITE 19.0%/28.9% (worst on both). Same strengths, same blind spots, on a generation run the model never saw.
- FX tier on LI: 69/78 positive (88.5%), spans all 8 typologies, 9 false positives — same character as on HI.
- Per-typology recalls on shared typologies run somewhat lower on LI (consistent with 2,122 vs 2,543 test positives and a lower base rate feeding CF less training signal).

## Base facts

- Data: `hi_small_0.parquet` (4,150,000 rows) + `hi_small_1.parquet` (928,345 rows) = 5,078,345 transactions.
- Account identity: `(Bank, Account)` pair, NOT `Account` alone — 4 accounts in file 0 collide across banks under `Account` alone.
- Accounts: 515,088. Positive (touched by ≥1 flagged transaction, either side): 6,357 = 1.2342%.
- Transaction-level positive rate: 5,177 / 5,078,345 = 0.1019%.

## Robustness + operational precision (phase2_robustness.py) — RUN THIS BEFORE TRUSTING ANY NEW HEADLINE

**The headline number is stable across seeds.** 5 seeds, CF rebuilt per seed from that seed's train labels:

```
AUROC:  mean 0.9130   std 0.0042   min 0.9074   max 0.9166
shuffle control: mean 0.4987
p@npos: mean 0.3610   std 0.0096   min 0.3500   max 0.3728
seed 0 alone (the previously reported headline): AUROC 0.9161, p@npos 0.3559
```

Seed 0 was typical, not a favourable draw. 0.9161 stands.

**But precision is INVERTED at the top of the ranking — the AUROC hides this completely.**

```
worklist size      precision    hits  reviewed      lift
top 100               4.25%    10.2     258.8      3.4x
top 500              13.52%    67.6     500.0     11.0x
top 1000             38.88%   388.8    1000.0     31.5x
top ~2543            36.10%   918.0    2543.0     29.2x
```

Worked out by band: ranks 1–259 hold ~11 positives (4%), ranks 260–500 hold ~57 (24%), ranks 500–1000 hold ~321 (**64%**). Precision *increases* going down the list. The model's most confident predictions are its worst.

**Cause — same bug as the explainability finding below.** Asking for top-100 returns 258.8 accounts: a tie block. Accounts whose only counterparty is the giant hub `70:100428660` (14,775 touchers) all receive an identical score, pile up at the very top of the ranking, and are overwhelmingly false positives. `phase2_explain.py` found this qualitatively; this run measures its operational cost.

**Why this matters more than the AUROC**: an investigator handed the top 100 gets a 4% hit rate; handed the top 1000, 39%. A single aggregate ranking metric gave no hint. Always report precision at operational k, not just k=n_positives.

## Hub tie-block fix (phase2_cf_hubexclude.py) — RESOLVED, with one stated trade-off

The top-of-ranking tie-block defect above is fixed by **explicit hub exclusion**: drop hub counterparty columns from the CF matrix entirely (steeper down-weighting had already failed — `phase2_cf_strongidf.py` — because row-normalization erases any IDF penalty for single-counterparty accounts, exactly the failing case).

**Cutoff chosen from population structure, no labels involved**: the counterparty-popularity distribution has a literal empty gap — 515,073 of 515,088 counterparties have popularity ≤ 93, none fall in 94–566, and exactly 15 (all bank 70, from 567 up to `70:100428660` at 14,775) sit above. Cutoff = exclude popularity > 100; any cutoff inside the gap selects the same 15 hubs. 1,590 hub-only accounts get an empty CF row and score 0.

**Result, mean across 5 seeds, both variants scored side by side in the same run:**

```
metric        baseline  hub_excluded     delta
auroc           0.9130        0.8987   -0.0143
p@100           0.0425        0.8660   +0.8235
p@500           0.1352        0.6940   +0.5588
p@1000          0.3888        0.5724   +0.1836
p@npos          0.3610        0.4112   +0.0503
sel@100          258.8         100.0   (tie block gone)
```

**Verdict against the criterion decided in advance** ("p@100 improves substantially; p@1000 and AUROC do not degrade"): p@100 4.25% → 86.60% (lift 3.4x → 70.2x), p@1000 and p@npos also improved — but **AUROC degraded 0.9130 → 0.8987**, so the criterion is not fully met and the trade-off must stay stated, not buried. Improvements held on every seed individually (hub_excluded p@100 range 0.83–0.89; baseline range 0.029–0.073). Shuffle controls clean both variants (0.4977 / 0.4987); ~201k distinct scores, no tie degeneracy; top-100 returns exactly 100 accounts on all 5 seeds.

**Why AUROC dips**: AUROC grades the ordering of the entire 206,036-account test list. Touching a giant hub was weakly informative deep in the ranking (cf-alone AUROC 0.8832 → 0.8600 after exclusion) but poison at the top. By this file's own operational standard (see robustness section: precision at operational k matters more than the aggregate), **hub_excluded is the operational recommendation**; the unexcluded 0.9161 model remains the AUROC-best bulk ranker. Untested follow-up if anyone wants the last AUROC back: use hub_excluded to build the worklist top and the baseline for deep ranking — two scores, two jobs.

## Explainability (phase2_explain.py)

`combined_with_cf`/`cf_damped` are not black boxes — cf_damped's score is a literal sum of contributions, traceable per account: which counterparties it shares with which specific training-positive accounts, and how rare/common each shared counterparty is. `phase2_explain.py` prints this trace for real examples.

**True positives** trace cleanly: small, specific shared counterparties (8–26 total touchers) shared with 2–3 known-bad training accounts. Legible, plausible reasoning.

**Found a real, fixable weakness**: all top-5 false positives traced to the *same single counterparty*, `70:100428660` — the largest hub in the dataset, touched by 14,775 accounts. With that many touchers, ~180 would be positive by pure base-rate chance, so touching it carries almost no signal, yet the sqrt-IDF damping wasn't suppressing it enough to stop it dominating scores for accounts whose only connection is that hub.

**Attempted fix, and why it failed** (`phase2_cf_strongidf.py`): tried linear IDF (`1/(1+popularity)`) instead of sqrt IDF (`1/sqrt(1+popularity)`) — 121.6x stronger penalty on the hub. Result: **worse overall** — AUROC 0.8768 vs 0.8832, precision@k 0.2694 vs 0.3405 (181 fewer correct hits). The hub-driven false-positive scores did drop (3.5414 → 1.0491) but not to zero, and the same steeper penalty over-penalized the small counterparties that were driving genuine true positives. Root cause: row-normalization erases any IDF penalty for accounts that touch only ONE counterparty (normalizing a single value to unit length always gives 1, regardless of the original weight) — exactly the case of the hub-only false positives, so the fix couldn't reach the actual problem. A more surgical fix (explicit hub exclusion above some popularity level, not just steeper damping) is untested. Two attempts at improving the base CF recipe (amount-weighting, steeper IDF) have now both come back negative — the original sqrt-IDF recipe may already be a reasonable local optimum.

## What's next (see MATH_IDEAS.md for the full untried-idea catalog)

**The hub tie-block fix is DONE** (`phase2_cf_hubexclude.py`, section above): p@100 4.25% → 86.60% across 5 seeds, tie block eliminated, at the cost of −0.0143 AUROC. Hub exclusion (popularity > 100, chosen from the empty 94–566 gap in the distribution, not the labels) is now the operational recommendation.

**2-hop CF tried and rejected** (`phase2_cf_2hop.py`, caveat above) — CF is a confirmed local optimum; stop extending it.

**Structural invariants tried** (`phase2_invariants.py`, caveat above) — the invariant idea itself is dead on this generator, but its audit produced the has_fx flag, now part of the operational best (0.7176 p@500 / 0.5912 p@1000, 5-seed means).

**Twenty questions applied** (`phase2_twentyq_worklist.py`, section above) — identification of individual suspects is capped by signature collisions (441/1000), but the audit produced a near-pure first triage tier (cross-currency ∩ worklist: 97.8% positive, n=270) and three cheap innocence markers.

**Transfer test run** (`phase3_transfer_li.py`, section above) — the recipe generalizes as "much better than chance" (55x lift@100) but not at its HI-Small headline level (AUROC 0.90 → 0.69). Highest-value open item: rerun on TRUE LI-Small (Kaggle manual download) to separate generator-overfit from subsample-starvation.

**Reprioritized 2026-08-07 after the typology ground-truth audit**: (1) ~~targeted detection of STACK and BIPARTITE~~ — **tried twice and closed** (butterflies, flow ratio — caveat above): those typologies are account-level-invisible by construction; further progress on them requires transaction-level scoring or a head-on, named decision about temporal features, not another aggregate feature. (2) Transfer audit of per-typology recall on true LI-Small (`LI-Small_Patterns.txt` already extracted). (3) HI-Medium as the next transfer rung (in `archive.zip`, ~31M rows — memory feasibility unknown). MATH_IDEAS.md's remaining untried list (Ricci, persistent homology, etc.) stays below these — elaborate-geometry and amount-weighted families keep losing here (flow ratio made it four amount-based negatives), while every win this session came from simple structural facts with a measured target.
