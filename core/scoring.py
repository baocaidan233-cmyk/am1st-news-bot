"""The scorer's closed-set answer shape and the code that turns it into a score.

Lives in core/ rather than next to either scorer because BOTH use it: the
production Scorer (agents/scorer.py) computes its score here, and the shadow
scorer that now runs the retired prompt alongside it reads the same weights when
reporting. Moved here from agents/scorer_shadow.py on 2026-10-02, when the two
scorers swapped places.

WHY THE MODEL IS NOT ASKED FOR A NUMBER
---------------------------------------
The retired prompt (prompts/scoring_prompt.txt, kept and still running in
shadow) asked gpt-4o-mini for a 1-10 score against a 21-item list of Core
Themes. Two things were wrong with that, both measured:

  * The theme list is a whitelist, so a story in no category is unscoreable
    rather than low-scoring. The clearest case: the Lindsay Clancy retrial, the
    single most-rejected story in the whole decision log — 66 candidates, 44
    scored exactly 4.0 against a 5.0 floor, none published, because an ordinary
    murder case is on no theme. A whitelist that misses a category loses it
    permanently; a blacklist that misses one costs one extra call.
  * Asked for a number, the model barely used the range. 7 of 3,049 peer posts
    scored >=8. Its real span was 2-6, and above its own floor it had two
    values, so 37% of our own published posts came out at exactly 6.0 and could
    not be ranked against each other at all.

So the model answers four judgments it can actually make and the code does the
arithmetic. The weights below are not guesses.

THE MEASUREMENTS (engagement = likes + comments + reposts, read at a fixed 12h
age, normalised by the median of its own UTC posting hour; peers additionally
normalised by their own channel's median):

    g non-empty      1.23x [1.15, 1.33]   ours
    d = stopped      1.16x [1.07, 1.23]   ours   1.59x [1.44, 1.75]  peers
    who = ally       1.21x [1.13, 1.35]   ours   1.52x [1.25, 1.79]  peers
    who = opponent   1.06x [1.01, 1.14]   ours   1.28x [1.19, 1.38]  peers
    who = ordinary   1.05x [0.97, 1.17]   not significant on either corpus
    d = happening    1.01x [0.98, 1.10]   zero — scores nothing, by design
    d = price        1.10x [0.81, 1.20]   NOT ESTABLISHED, n=24. The 0.5 below
                                          is a placeholder. Do not cite it.

g is deliberately free text with no list to choose from. The same question
asked as an 11-item enumeration measured WEAKER (1.12x vs 1.23x) — enumerating
it recreated in miniature the whitelist problem it was meant to solve. 14 days
of our own published posts produced 472 distinct grievance strings out of 507,
and the 7% that repeat a phrase carry no more engagement than the unique ones
(1.03x vs 1.05x), so the repetition is harmless noise rather than the signal.

Two further dimensions — how decisive the action was, and whether the story is
the first instalment or the Nth — each looked strong alone (1.43x and 1.30x on
peers) and were REJECTED: inside a fixed (d, who) cell they add nothing (1.10x),
the head-to-head with and without them sits inside the noise, and asking for
them shifted the four existing fields on 7-11% of items. They were measuring the
same underlying thing twice. Do not re-add either without a within-cell test.

WHAT THIS LADDER IS WORTH, measured on 716 of our own published posts over 14
days and on 900 posts from four peer MAGA news channels (a corpus nothing here
was fitted on), against the prompt it replaces:

                        retired prompt      this
    top 25% / bottom 25%    1.16x          1.29x    ours
    top 25% / bottom 25%    1.18x          1.38x    peers
    highest / lowest band   1.38x          1.62x    ours
    largest tied band         37%            25%
"""

from __future__ import annotations

import re

_D = {"stopped", "price", "happening", "none"}
_WHO = {"opponent", "ally", "ordinary", "none"}
_DROP = {"anti_admin", "trivia", "none"}

# Channel duty, not a fitted weight: the standing instruction (2026-10-01) is
# that China, Xi, Taiwan and CCP influence are part of what this channel exists
# to cover and are not to be cut for low measured engagement — AM1ST's China
# topic measures 0.83x and stays regardless. Implemented as a FLOOR so it can
# only ever raise a score, never reject: a keyword this regex misses costs that
# story some rank, never its life. That is the whole reason it is not a gate.
_CHINA = re.compile(
    r"\b(china|chinese|ccp|xi jinping|beijing|taiwan|taipei|hong kong|huawei)\b", re.I)
_CHINA_FLOOR = 5.5


def compute_score(fields: dict, text: str = "") -> float:
    """Every number here is from the module docstring. Nothing is a guess except
    the 0.5 on d=price, which is labelled as such in both places."""
    if fields["drop"] != "none":
        return 0.0
    s = 4.0
    if fields["g"] != "none":
        s += 1.5
    if fields["d"] == "stopped":
        s += 1.0
    elif fields["d"] == "price":
        s += 0.5          # placeholder, n=24, CI crosses 1.0
    if fields["who"] == "ally":
        s += 1.0
    elif fields["who"] in ("opponent", "ordinary"):
        s += 0.5
    if _CHINA.search(text or ""):
        s = max(s, _CHINA_FLOOR)
    return min(s, 9.0)


def normalise(d) -> dict | None:
    """The model answers 'none' in whichever field has nothing in it, so every
    field's empty value is spelled 'none' and the two it tends to spell
    differently are mapped back. Anything still outside the closed sets is a
    parse failure, not something to guess at.

    This mapping is not cosmetic. The first draft used 'none' for g and drop but
    'neither' for d and 'no' for who, and the model filled 'none' into all four:
    152 of 500 answers were unusable. One spelling everywhere plus this mapper
    took that to 1 in 500."""
    if not isinstance(d, dict):
        return None
    direction = (d.get("d") or "none").strip().lower()
    if direction == "neither":
        direction = "none"
    who = (d.get("who") or "none").strip().lower()
    grievance = (d.get("g") or "none").strip()
    if grievance.lower() in ("none", "n/a", ""):
        grievance = "none"
    drop = (d.get("drop") or "none").strip().lower()
    if direction not in _D or who not in _WHO or drop not in _DROP:
        return None
    return {"g": grievance, "d": direction, "who": who, "drop": drop}


def describe(fields: dict) -> str:
    """One line for the Notion llm_comment column, so a human reviewing the pool
    sees WHY a candidate scored what it did rather than a bare number."""
    bits = [f"d={fields['d']}", f"who={fields['who']}"]
    if fields["drop"] != "none":
        bits.append(f"DROP={fields['drop']}")
    g = fields["g"]
    return ("; ".join(bits) + (f" | {g}" if g != "none" else " | 无诉求"))[:400]
