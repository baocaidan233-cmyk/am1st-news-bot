"""English-only language guard — AM1ST is an English-only channel, but
nothing previously verified that either a candidate's own title/description
or its underlying article text was actually English. Added 2026-08-06 after
a real published post turned out to be a Portuguese-language Reuters article
(reuters.com/pt/...) that the writer tried, and failed, to self-censor via
a "No comment" reply — relying on the writer's own judgment alone isn't
robust enough; this is an explicit, code-level check that doesn't depend on
the model noticing and correctly signaling it."""

from __future__ import annotations

import logging

from langdetect import DetectorFactory, LangDetectException, detect

logger = logging.getLogger(__name__)

# Deterministic detection — langdetect's default behavior draws randomly
# from character n-gram probabilities, which can give a different answer
# for the same text between runs. A fixed seed makes is_english() a pure
# function of its input, which matters for anything that logs/tests it.
DetectorFactory.seed = 0

_MIN_LENGTH = 20


def is_english(text: str) -> bool:
    """Fails open (returns True) on empty/very short text or a detection
    error — language detection is unreliable on a handful of words, and
    wrongly rejecting a real English candidate over a short title is worse
    than occasionally letting a short non-English one slip through (later
    pipeline stages, e.g. the writer, still have a chance to catch it)."""
    cleaned = text.strip()
    if len(cleaned) < _MIN_LENGTH:
        return True
    try:
        return detect(cleaned) == "en"
    except LangDetectException:
        return True
