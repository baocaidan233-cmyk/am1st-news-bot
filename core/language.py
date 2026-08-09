"""English-only language guard — AM1ST is an English-only channel, but
nothing previously verified that either a candidate's own title/description
or its underlying article text was actually English. Added 2026-08-06 after
a real published post turned out to be a Portuguese-language Reuters article
(reuters.com/pt/...) that the writer tried, and failed, to self-censor via
a "No comment" reply — relying on the writer's own judgment alone isn't
robust enough; this is an explicit, code-level check that doesn't depend on
the model noticing and correctly signaling it.

Extended 2026-08-09 with a CJK-ratio pre-check: langdetect scores a whole
string's dominant language by character n-gram statistics, so a Chinese
headline translation concatenated with its (long) English original —
e.g. "习惯高油价吧 - Get Used to High Gas Prices" — reads as majority-English
and passes detect() as "en" even though the leading clause is genuinely
Chinese. Checked against 3547 real historical candidates: every case with
any Han/Kana/Hangul content at all landed at a CJK-letter ratio of 0.133
or higher, and every pure-English candidate landed at exactly 0 — no
observed case in between — so the 0.05 threshold below has wide margin on
both sides of that real split."""

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

# CJK Unified Ideographs, Extension A, Compatibility Ideographs, Hiragana/
# Katakana, Hangul syllables — covers Chinese/Japanese/Korean, the scripts
# actually observed slipping through langdetect in AM1ST's real feed.
_CJK_RANGES = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0xF900, 0xFAFF),
    (0x3040, 0x30FF),
    (0xAC00, 0xD7A3),
)
_CJK_RATIO_THRESHOLD = 0.05


def _is_cjk_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _cjk_ratio(text: str) -> float:
    """Ratio computed over alphabetic characters only — digits/punctuation
    carry no language signal (a dollar figure or a date reads the same in
    any language) and would only dilute the ratio in either direction."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    cjk = sum(1 for ch in letters if _is_cjk_char(ch))
    return cjk / len(letters)


def is_english(text: str) -> bool:
    """Fails open (returns True) on empty/very short text or a detection
    error — language detection is unreliable on a handful of words, and
    wrongly rejecting a real English candidate over a short title is worse
    than occasionally letting a short non-English one slip through (later
    pipeline stages, e.g. the writer, still have a chance to catch it).

    The CJK check runs before the length fail-open on purpose: a short
    string can still be a complete, information-dense Chinese clause (CJK
    characters carry more per-character meaning than Latin letters), so
    "short" shouldn't imply "safe to wave through" the way it does for
    langdetect's Latin-script blind spot below."""
    cleaned = text.strip()
    if _cjk_ratio(cleaned) > _CJK_RATIO_THRESHOLD:
        return False
    if len(cleaned) < _MIN_LENGTH:
        return True
    try:
        return detect(cleaned) == "en"
    except LangDetectException:
        return True
