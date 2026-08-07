# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 James Hawkins / Komposos-Labs

"""
Twenty questions against the graph: identify the adversary without scanning it.

The problem with a billion edges
--------------------------------
You cannot look at all of it, and you should not want to. Whole-graph scanning
answers "what is out there"; an incident responder has a different question —
"who is in my network, and what will they do next" — and that one is answered
by asking, not by sweeping.

Information theory gives the bound. ATT&CK describes 944 catalogued profiles (707 malware families, 166 named
groups, 71 tools) with two or
more techniques, and log2(944) is about 9.9, so **ten perfectly chosen
yes/no questions are enough to name one**. Twenty is generous. The whole game
is choosing the questions.

How a question is chosen
------------------------
A question here is "have you seen technique T?". Asking it splits the surviving
candidate actors into those that use T and those that do not. The value of the
question is the expected entropy it removes:

    H(S)        = log2(|S|)                          — uniform over survivors
    IG(T)       = H(S) - [ p·H(S_yes) + (1-p)·H(S_no) ]

Maximised when the split is even. A technique used by every actor teaches
nothing (everyone survives); one used by nobody teaches nothing either. The
best question is the one you genuinely cannot predict the answer to — which is
also why "ask about the most common technique" is a bad strategy, and why this
beats the obvious baseline.

What this is for
----------------
1. **Incident response.** "Check for T1021 next" — chosen because it splits the
   hypothesis space, not because it is popular.
2. **Using a large graph engine sanely.** Twenty targeted queries against a
   billion-edge graph is a different cost class from a whole-graph sweep, and
   an engine like xGT answers each one as a bounded pattern match.

Honest scope: this identifies which ATT&CK actor profile best matches observed
technique usage. That is not the same as attribution — several actors share
technique sets, and real intrusions include techniques nobody logged.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def entropy(n: int) -> float:
    """Bits needed to name one of n equally likely candidates."""
    return math.log2(n) if n > 1 else 0.0


@dataclass
class Question:
    technique: str
    information_gain: float
    splits_to: Tuple[int, int]      # (candidates if yes, if no)

    def __str__(self) -> str:
        yes, no = self.splits_to
        return (f"seen {self.technique}?  gain {self.information_gain:.3f} bits "
                f"({yes} yes / {no} no)")


@dataclass
class Answer:
    technique: str
    observed: bool


class TwentyQuestions:
    """Narrow a candidate set by asking the most informative question next."""

    def __init__(self, actor_techniques: Dict[str, Set[str]]):
        self.actors: Dict[str, Set[str]] = {
            a: set(t) for a, t in actor_techniques.items() if len(t) >= 2
        }
        self.by_technique: Dict[str, Set[str]] = defaultdict(set)
        for actor, techs in self.actors.items():
            for t in techs:
                self.by_technique[t].add(actor)
        self.candidates: Set[str] = set(self.actors)
        self.asked: List[Answer] = []

    # -- the game ---------------------------------------------------------

    def bits_remaining(self) -> float:
        return entropy(len(self.candidates))

    def best_question(self, exclude: Optional[Set[str]] = None
                      ) -> Optional[Question]:
        """The technique whose answer removes the most uncertainty."""
        exclude = exclude or {a.technique for a in self.asked}
        n = len(self.candidates)
        if n <= 1:
            return None

        h_now = entropy(n)
        best: Optional[Question] = None
        for technique, users in self.by_technique.items():
            if technique in exclude:
                continue
            yes = len(users & self.candidates)
            no = n - yes
            if yes == 0 or no == 0:
                continue            # tells us nothing: everyone answers alike
            p = yes / n
            h_after = p * entropy(yes) + (1 - p) * entropy(no)
            gain = h_now - h_after
            if best is None or gain > best.information_gain:
                best = Question(technique, gain, (yes, no))
        return best

    def answer(self, technique: str, observed: bool) -> int:
        """Record an answer and prune. Returns survivors remaining."""
        users = self.by_technique.get(technique, set())
        self.candidates = (self.candidates & users if observed
                           else self.candidates - users)
        self.asked.append(Answer(technique, observed))
        return len(self.candidates)

    def ranked_candidates(self, top_k: int = 5) -> List[Tuple[str, int]]:
        """Survivors, smallest repertoire first — the tightest explanation
        of what was seen is preferred over an actor that does everything."""
        return sorted(((a, len(self.actors[a])) for a in self.candidates),
                      key=lambda kv: kv[1])[:top_k]


# ---------------------------------------------------------------------------
# Baselines and evaluation
# ---------------------------------------------------------------------------

def play(actor_techniques: Dict[str, Set[str]], truth: str,
         strategy: str = "information", max_questions: int = 20
         ) -> Tuple[int, int]:
    """Play against a known actor. Returns (questions asked, survivors left).

    The oracle answers truthfully from that actor's real technique set.
    """
    game = TwentyQuestions(actor_techniques)
    truth_techniques = set(actor_techniques[truth])
    popularity = sorted(game.by_technique,
                        key=lambda t: -len(game.by_technique[t]))

    asked: Set[str] = set()
    for step in range(max_questions):
        if len(game.candidates) <= 1:
            break
        if strategy == "information":
            q = game.best_question(exclude=asked)
            technique = q.technique if q else None
        elif strategy == "popularity":
            technique = next((t for t in popularity if t not in asked), None)
        else:
            raise ValueError(strategy)
        if technique is None:
            break
        asked.add(technique)
        game.answer(technique, technique in truth_techniques)

    return len(game.asked), len(game.candidates)


def evaluate(actor_techniques: Dict[str, Set[str]], sample: int = 60,
             seed: int = 0) -> Dict[str, Dict[str, float]]:
    import random
    rng = random.Random(seed)
    pool = [a for a, t in actor_techniques.items() if len(t) >= 4]
    chosen = rng.sample(sorted(pool), min(sample, len(pool)))

    out: Dict[str, Dict[str, float]] = {}
    for strategy in ("information", "popularity"):
        asked_counts, solved = [], 0
        for actor in chosen:
            n_asked, survivors = play(actor_techniques, actor, strategy)
            asked_counts.append(n_asked)
            if survivors <= 1:
                solved += 1
        out[strategy] = {
            "mean_questions": sum(asked_counts) / len(asked_counts),
            "identified_within_20": solved / len(chosen),
        }
    return out


def main() -> None:
    import argparse

    from cyber.cooccurrence_predictor import (
        ACTOR_TYPES, actor_composition, load_actor_techniques,
    )

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--types", nargs="+", default=list(ACTOR_TYPES),
                    choices=list(ACTOR_TYPES),
                    help="which ATT&CK objects count as a candidate "
                         "(default: all three)")
    args = ap.parse_args()

    actors = load_actor_techniques(args.types)
    usable = {a: t for a, t in actors.items() if len(t) >= 2}
    if len(usable) < 2:
        print("fewer than two candidates; nothing to play against")
        return

    # Never print a candidate count without its composition. Reporting 944 as
    # a count of groups, when 707 of them are malware, is the one error this
    # module has actually made in front of someone.
    composition = actor_composition(usable)
    breakdown = ", ".join(f"{n} {t}" for t, n in sorted(composition.items(),
                                                        key=lambda kv: -kv[1]))
    print(f"candidate profiles: {len(usable)}  ({breakdown})")
    print(f"theoretical minimum: {entropy(len(usable)):.1f} perfect questions")
    print()

    results = evaluate(usable)
    for strategy, stats in results.items():
        print(f"  {strategy:<12} mean questions {stats['mean_questions']:.1f}   "
              f"identified within 20: {stats['identified_within_20']:.0%}")

    print()
    print("Example game (first five questions against the full actor set):")
    game = TwentyQuestions(usable)
    asked: Set[str] = set()
    for _ in range(5):
        q = game.best_question(exclude=asked)
        if q is None:
            break
        print(f"  {game.bits_remaining():5.1f} bits left — {q}")
        asked.add(q.technique)
        game.answer(q.technique, True)


if __name__ == "__main__":
    main()
