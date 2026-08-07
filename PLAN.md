# KOMPOSOS-AML — plan

**Started 2026-08-07, after closing KOMPOSOS-SEC. Read `WHY.md` before
adding anything.**

## The one rule, carried over

**Runs are truth.** Every number quoted must come with the command that
printed it, in the session that quotes it. Anything from a previous session is
marked and re-run before it goes in front of anyone.

## Why this repo exists, in one paragraph

KOMPOSOS-SEC spent a year building detection methods that could not be scored,
because the labelled data arrived last. When it finally arrived, the best
method put the attacker's machine 104th of 15,683. **This repo inverts that
order: score first, build second.** The IBM AML dataset ships a per-transaction
`Is Laundering` label, so every idea is checkable on day one.

## Why AML and not more security

Two structural reasons, both measured rather than hoped:

1. **Twenty questions needs an enumerated candidate set with known features.**
   "Is there an intruder in my network" has no candidate list, which is exactly
   why the method could not find hidden actors in KOMPOSOS-SEC. Laundering
   *typologies* are a finite catalogued list — layering, round-robin through
   shells, mule ring, shared identifiers across accounts. Same shape as the 166
   ATT&CK groups the method does work on.
2. **The AML patterns are genuinely graph shapes.** A round-robin is a cycle. A
   mule network is a fan. Layering is a path. Unlike authentication logs, which
   turned out to be a counting problem with no graph in them.

## Phase 0 — RUN 2026-08-07. ANSWERED. Read this before downloading anything.

**Rocketgraph's AML dataset has NO LABEL. Do not use it.** Measured by
`phase0_inspect.py` against their 1M subset:

```
TRANSACTIONS (edge frame) — 1,000,000 rows
   from_account_id  to_account_id  timestamp
   amount_paid      amount_received
   paid_currency    received_currency  payment_type

ACCOUNTS (vertex frame) — 1,101,709 rows
   acct_id  bank_number  account_number

LABEL CHECK: no column matched any label hint.
```

They dropped `Is Laundering` when restructuring IBM's file into vertex and
edge frames. Eight columns, no ground truth. The files are in `data/` if
anything ever needs the graph structure, but **nothing can be scored on them**.

This is the second time Rocketgraph's hosted data has been the unlabelled
variant — the first was the 2017 LANL set rather than the labelled 2015 one.
They publish data shaped for their engine, and a label is not graph structure.
**Assume any dataset they host is unlabelled until proven otherwise.**

### The labelled source — FOUND, and it needs no account

Kaggle hosts IBM's original but requires a login. A **Hugging Face mirror of
HI-Small does not**, and it carries the label:

```
eexzzm/IBM-Transactions-for-Anti-Money-Laundering-HI-Small-Trans

5,078,345 rows, ~91 MB as parquet
Timestamp, From Bank, Account, To Bank, Account.1,
Amount Received, Receiving Currency, Amount Paid,
Payment Currency, Payment Format, Is Laundering    <-- binary 0/1
```

Direct download (no auth):

```
https://huggingface.co/api/datasets/eexzzm/IBM-Transactions-for-Anti-Money-Laundering-HI-Small-Trans/parquet/default/train/0.parquet
https://huggingface.co/api/datasets/eexzzm/IBM-Transactions-for-Anti-Money-Laundering-HI-Small-Trans/parquet/default/train/1.parquet
```

Saved as `data/hi_small_0.parquet` and `data/hi_small_1.parquet`.

HI = higher illicit ratio, and Small = fastest to falsify an idea against.
That is why this variant and not the others. The full set (HI/LI x
small/medium/large) is on Kaggle if a bigger one is ever needed:
`kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml`

### Still to answer once the file is on disk

1. Confirm `Is Laundering` is present after the parquet conversion.
2. Class balance and the base rate.
3. Basic shape — accounts, banks, currencies, time span.

**If the label turns out to be absent there too, stop and reconsider the whole
plan.** Everything below depends on being able to score.

## Phase 1 — the baseline comes FIRST

This is the inversion. KOMPOSOS-SEC built elaborate methods and scored them
last; every one lost to counting. So start with counting and write the number
down before anything else exists to compare against.

Simplest possible features per account, all pure arithmetic:

- number of counterparties (degree)
- total amount in / out, and the ratio
- number of transactions
- distinct banks touched

Score each as a ranker against the label. Record: AUROC or precision@k, plus
**a random baseline in the same run**, plus the **class balance** (with a 0.1%
positive rate, accuracy is meaningless and precision@k is the honest measure).

**Deliverable: a table of counting baselines with numbers. Nothing else.**

## Phase 2 — only now, try something structural

A structural method has to beat the Phase 1 table *under a control*. The
controls that caught real errors in KOMPOSOS-SEC, all of which must be
standard here:

- **Negative control.** Shuffle the labels (or the edges, degree-preserved) and
  re-run. A method that still wins on structureless data is measuring an
  artefact. This is what killed `hunt_targets`.
- **Confound control.** If candidates are drawn at random they are mostly
  unconnected, and any structural method scores well for detecting connectivity
  rather than predicting. Draw decoys from the *reachable* set. This turned two
  apparent winners into below-chance results.
- **Distinct-score reporting.** Always print how many distinct scores a method
  produced. A universal tie can read as a perfect ranking. It did once.
- **Ties take the average rank**, never "count of strictly higher".

Candidate structural ideas, in order of how cheap they are to falsify:

1. **Cycle detection.** Round-robin laundering is literally a cycle. Count
   cycles each account participates in. Cheap, and it is a graph operation.
2. **Twenty questions over typologies.** Needs a typology → feature matrix that
   does not exist yet; building it is the real work, and it must not be built
   by looking at the labels.
3. **Anything categorical.** Deliberately last. Two independent measurements in
   KOMPOSOS-SEC found elaborate structure losing to arithmetic. That is not a
   prohibition, it is a prior — see `WHY.md`.

## What was ported, and what deliberately was not

**Ported:**
- `twenty_questions.py` — unmodified. It takes `Dict[str, Set[str]]` and knows
  nothing about ATT&CK, so it works on any candidate/feature population.

**To vendor when needed, not before:**
- `core/evidence_gate.py` + `vendor/haloa` + `vendor/nlock` from KOMPOSOS-SEC —
  refuses to emit a finding without its route and trigger, and permanently
  stamps a result `simulated` if the input declared its own answer.

**NOT ported, on purpose:** everything in `categorical/`, `zfc/`, `oracle/`,
`geometry/`, `topology/`, `hott/`, `cubical/`, `game/`. All measured losing to
counting on the one task they were ever scored on. They are one `git clone`
away if a phase-2 idea genuinely needs them.

## The testing standard, carried over unchanged

Every detection test must:

1. use input carrying **no** annotation of what should be found; and
2. have a **negative counterpart** — the same shape without the property. A
   tool that flags everything passes all the positives.

## Things that will bite you

- **Class imbalance.** Laundering is rare. Accuracy will look superb and mean
  nothing. Use precision@k and always print the base rate.
- **Synthetic data.** This is generated by a multi-agent simulation, not real
  banking. A method that works here may be learning the generator. Say so.
- **Window choice is a threshold in disguise.** Picking a time window until the
  signal appears is how two KOMPOSOS-SEC detectors died. Fix the window in
  advance and hold out a second occurrence to check it.
