# Why this repo starts the way it does

**Written 2026-08-07 from measured results in KOMPOSOS-SEC. This is not
philosophy. Every claim below is a number that was printed by a command.**

Read this before adding a method. It exists so the year that produced it does
not have to be repeated.

## 1. The order of work was wrong, and that was the whole problem

KOMPOSOS-SEC built detection methods for a year and scored them at the end.
The ground-truth labels arrived on the final night. The first scoring run
killed the best method:

```
1,051,430,459 authentication events, 15,683 source hosts

  C17693         117 accounts   rank 104 of 15,683
  C18025          17 accounts   rank 512 of 15,683
  C19932           2 accounts   rank 7874 of 15,683
  C22409           7 accounts   rank 1362 of 15,683
```

C17693 was responsible for 701 of the 749 labelled attacker logins. It ranked
104th. **Build second. Score first.**

## 2. A separation over a short window can vanish over a long one

The method above looked excellent on **one day** of data — median 1 account per
machine, only 4 machines above 20. Over **58 days** the background accumulated
and the signal did not: median 2, 401 machines above 20, 123 above 100.

Nothing about the attacker changed. The reference population did.

## 3. Elaborate structure lost to arithmetic, twice, independently

Scored on ATT&CK, leave-one-out, n=40, chance MRR 0.0952, with the
reachability confound controlled:

```
~cooccurrence           0.4375      (collaborative filtering — counting)
~popularity             0.3995      (how common a technique is — counting)
YonedaPattern           0.3532
Geometric               0.2385
FibrationLift           0.0980
ToposLogic              0.0952      one distinct score across 20 candidates
OperadicDecomp          0.0952      one distinct score across 20 candidates
KanExtension            0.0938      one distinct score across 20 candidates
CubicalGapFill          0.0817      below chance
Composition             0.0817      below chance
```

Independently, `attack_nerve_eval` measured a horn construction at 0.696 AUROC
against a raw `degree_product` baseline at 0.906.

**This is a prior, not a prohibition.** The diagnosis is specific and testable:
ATT&CK is an *unweighted* graph, so a weighted composite along a path
degenerates into counting. The AML transaction graph **has weights** — amounts,
timestamps, counts. That is the first honest test of the diagnosis, and it is
the reason this repo is worth starting.

But it is phase 2, after a counting baseline exists to beat.

## 4. Three ways a harness lies, all of which happened

**A universal tie reads as a perfect score.** The first scoring attempt
returned MRR 1.0000. Every candidate had scored exactly 0.000, and rank was
computed as "how many score strictly higher", so everything ranked first.

> Always print the count of **distinct** score values. Ties take the average
> rank.

**Random decoys manufacture winners.** Decoys drawn at random are mostly
unconnected to the source; the hidden answer usually is connected. Anything
detecting connectivity then scores well without predicting anything:

```
                RANDOM    MATCHED     drop
Composition     0.3585     0.0817   -0.2768
CubicalGapFill  0.3585     0.0817   -0.2768
```

> Draw decoys from the reachable set, and report both pools.

**A method can beat its baseline on structureless data.** `hunt_targets` scored
0.694 vs 0.467 on real data — and 0.626 vs 0.320 on *shuffled* data, beating
the baseline by **more** once all real structure was destroyed.

> Every method gets a shuffle control. A win that survives the shuffle is not a
> win.

## 5. Silence is worse than an error

Four defects were found in KOMPOSOS-SEC code that everyone believed worked.
Three of them failed **silently**:

- A structural verifier built an undirected graph, so a real 3-hop path and its
  non-existent reverse both scored 0.40.
- A "drop-in replacement" for curvature matched the return type but not the
  interface. The caller wrapped it in `try/except`, so substituting it returned
  **nothing** instead of raising — indistinguishable from "found nothing".
- A ZFC verification adapter had **never executed**, because two files were
  written against different versions of the same API. It returned empty
  disagreement lists, which reads exactly like "both engines agree".
- A strategy emitted `inf` as a confidence and 15,274 predictions for a single
  query.

> Never swallow an exception around a scoring path. Assert that a method
> produced varied output before believing its number.

## 6. Methods fail because the data lacks the information, not because the
## mathematics is weak

Everything that failed in KOMPOSOS-SEC needed something the logs did not carry:

- twenty questions needs observed techniques — logs have no techniques
- activity theory needs motive and role — logs have no motive
- game theory needs payoffs — logs have no payoffs

An authentication log contains time, users, hosts, type, success. That is all.

> Before proposing a method, name the column it reads. If that column does not
> exist in the data, the method cannot work, however good it is.

## 7. What actually worked, in both repos so far

**Counting.** Information gain over a candidate set. Co-occurrence counts.
Distinct-account counts. An `if` statement that found a real misconfiguration.

The one clean result of the year was twenty questions: **7.5 questions against
an information-theoretic floor of 7.4**, beating the obvious alternative's
11.4, and passing its shuffle control. That is ported here.
