from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

# The last gate on editorial stance, on the exact text about to go out under
# this channel's name -- the caption and the headline card title together.
#
# Why it exists. On 2026-10-05 this channel published a Daily Beast story
# whose card title read "Trump Goons Melt Down at CNN Star Over Propaganda
# Ads" and whose caption called the administration's own ads "propaganda
# ads" in its own voice, then quoted a senator calling them "similar to
# authoritarian propaganda" with no answer from the administration anywhere.
# It took 26 engagement against a channel median near 60, so the editorial
# rule and the numbers agreed. Nothing in production could have stopped it:
# the anti_admin field lives in core/scoring.py, which the 10-02 revert left
# to the offline trainer only, and the editor prompt's one sentence about
# framing failed on 3 of its first 9 picks.
#
# Why not more prompt text. That sentence IS the prompt fix, and it did not
# hold. Three times on this project a rule added to a long prompt did nothing
# while the same question asked as its own narrow call worked -- same-event
# detection, the want gate, the subject closed set. So this is a separate
# call on a short fixed question, and the decision lands in code.
#
# What it must NOT block, because these are the channel's core material:
#   - the media, Democrats or an agency attacking Trump or the movement, the
#     attack reported as the news. "NYT tries to pin Cornell on Trump" is a
#     media-exposed story, not an anti-Trump one.
#   - a critic quoted when the administration's own answer is also in the
#     text. On 10-05 a Guardian piece ran here as "The White House defends
#     new press aide ... praised Harmony for being a trusted voice for the
#     MAGA movement" and took 85 engagement. The source is not the stance.
#   - criticism of a policy from the right: a RINO fight, a base revolt.
#   - enforcement, ICE and the Border Patrol doing their job, however rough
#     the footage.
#
# What it blocks: a hostile characterisation carried in the channel's OWN
# voice -- the caption's own words rather than a quote it attributes, or an
# epithet in the card title, which the reader sees before any caption.

# Epithets that cannot be a neutral description of the administration. Code
# rather than the model, because they are unambiguous and free to check, and
# a card title is short enough that proximity is real adjacency. A blocklist,
# so a word missing from it costs one model call, not a published post.
_EPITHET = (r"(?:goons?|thugs?|cronies|stooges|lackeys|henchmen|minions"
            r"|sycophants|regime|junta|cult)")
_SUBJECT = r"(?:trump|maga|white\s+house|administration|gop|republican)"
_EPITHET_NEAR_SUBJECT = re.compile(
    rf"\b(?:{_SUBJECT})\W{{0,12}}{_EPITHET}\b|\b{_EPITHET}\W{{0,12}}(?:{_SUBJECT})\b",
    re.I,
)

# The model names a target from a closed set and quotes the words. It does
# NOT decide whether the channel should publish -- that is code, below. The
# first version asked it the publish question directly and, over 22 real
# posts x 3 runs, blocked 3 good ones (85, 69 and 82 engagement, all above
# the channel median): a Guardian piece whose "sparked controversy" it read
# as our own sneer, an attack on AOC and Schumer it scored as an attack on
# the administration, and one of our own investigative pieces. Every miss
# was the same shape -- it flagged negative LANGUAGE rather than negative
# language AIMED AT US -- and all three were written into that prompt as
# must-pass examples before the run. So the judgement moved to code and the
# model kept the extraction, the job it does reliably here.
#
# The redesign was then run over all 1,124 posts this channel published from
# 2026-09-19, with their real engagement. It flagged 106 of them, and of the
# twelve highest-engagement flags about nine are plainly wrong in the same
# direction-blind way: "the prior administration relied on" (an attack on
# Biden), "echoing the administration commitment under President Trump to
# dismantle terror finance networks" (praise), "prompting ICE to defend its
# position" against rioters, "The Trump Effect on Immigration Is Real", and
# an ICE order that "Sparks Base Revolt" -- criticism from the right, which
# the prompt names as must-pass. It matches the WORD administration or ICE
# near anything negative, whichever way the negative points.
#
# The flagged group does run at 0.95x against 1.01x, but that is not evidence
# the stance call works. Split by target: opposition 1.07x with a 6.1%
# breakout rate, none 0.92x/3.8%, administration 0.92%/2.3%. The pattern is
# that NOT attacking the opposition underperforms -- the appeal-layer result
# arrived at through a different question -- and the blocked group is mostly
# posts that happen not to attack anyone.
#
# So this function stays unwired until there is a real set of violations to
# measure precision against. Two in 1,124 is not that set.
_TARGETS = ("administration", "enforcement", "opposition", "foreign", "none")
# The only two targets this channel cannot be hostile to in its own voice.
_BLOCKING = frozenset(("administration", "enforcement"))

_PROMPT = """You are reading text a news channel is about to post. Find the most hostile
characterisation the text makes IN ITS OWN VOICE -- its own wording, not a
claim it attributes to someone it names -- and say who that hostility is
aimed at.

"target" must be exactly one of:
  administration  President Trump, his officials, his appointees, his family,
                  the White House, or the MAGA movement itself
  enforcement     ICE, CBP, the Border Patrol, police, or a deportation
  opposition      Democrats, the left, socialists, activist groups, NGOs,
                  the mainstream media, a hostile court or agency, a RINO
  foreign         a foreign government, party, army or figure
  none            the text makes no hostile characterisation in its own voice

Reporting that someone attacked a target is not the text being hostile to
that target: "NYT blames Trump for Cornell" aims its hostility at the NYT.
A quote attributed to a named speaker is that speaker's voice, not the text's.

Reply with JSON and nothing else. The quote must be copied VERBATIM from the
text -- it is checked against it:

{"target": "opposition", "quote": "the exact words, copied"}
or
{"target": "none", "quote": ""}"""


def epithet_violation(caption: str, card_title: str = "") -> str | None:
    """The free half: an epithet about the administration, in our own text."""
    for where, text in (("card", card_title or ""), ("caption", caption or "")):
        m = _EPITHET_NEAR_SUBJECT.search(text)
        if m:
            return "epithet-in-%s:%s" % (where, m.group(0)[:48])
    return None


def _found(quote: str, *texts: str) -> bool:
    """Is the model quoting our text, or writing its own summary of it?"""
    q = " ".join((quote or "").lower().split())
    if len(q) < 8:
        return False
    for t in texts:
        if q in " ".join((t or "").lower().split()):
            return True
    return False


async def stance_violation(client, model: str, caption: str,
                           card_title: str = "") -> str | None:
    """None to publish, a rule name to drop.

    Fails open on every unknown: a missing caption, an API error, an
    unparseable reply, or a quote that is not in the text. An unverifiable
    verdict is not a verdict, and this gate sits in front of the only thing
    the channel produces."""
    rule = epithet_violation(caption, card_title)
    if rule:
        return rule

    if not caption or len(caption.strip()) < 40:
        return None

    body = caption if not card_title else (
        "CARD TITLE: %s\n\nCAPTION:\n%s" % (card_title, caption))
    try:
        r = await client.chat.completions.create(
            model=model, temperature=0, max_tokens=120,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": _PROMPT},
                      {"role": "user", "content": body[:2000]}],
        )
        d = json.loads(r.choices[0].message.content or "{}")
    except Exception:
        logger.debug("stance_guard: check failed, failing open", exc_info=True)
        return None

    if not isinstance(d, dict):
        return None
    target = str(d.get("target") or "none").strip().lower()
    if target not in _BLOCKING:
        return None
    quote = str(d.get("quote") or "")
    if not _found(quote, card_title, caption):
        logger.info("stance_guard: discarded %s verdict -- quote %r is not in the text",
                    target, quote[:60])
        return None
    return "hostile-at-%s:%s" % (target, quote[:64])
