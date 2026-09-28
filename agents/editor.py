from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel

from agents.topic_tagger import TOPICS
from agents.want_tagger import WANTS
from core.config import AppConfig
from core.models import PublishCandidate
from core.openai_client import create_openai_client

logger = logging.getLogger(__name__)


_SAME_EVENT_PROMPT = """These headlines were all chosen for the same feed. Which of them report the
SAME underlying news event?

One event, not two: an incident and a statement, reaction or response about
that same incident; the same ruling covered by two outlets; a story and its own
follow-up. Different events: two separate incidents, even similar ones, and two
different people acting on the same broad topic.

Return the numbers of every group of two or more that share an event. A
headline in no group is left out entirely.

Reply with JSON in exactly this form:
{"groups": [[2, 6]]}   — or {"groups": []} when they are all distinct."""


class EditorPick(BaseModel):
    candidate: PublishCandidate
    subject: str | None = None
    want: str | None = None
    why: str = ""

    class Config:
        arbitrary_types_allowed = True


class EditorPicker:
    """One call per publish cycle that reads the whole shortlist at once and
    returns the stories worth posting, ranked, each with a reason.

    Replaces a chain of independent per-story judgements. That chain was
    measured on 2026-09-28 and every link in it had collapsed: llm_score
    correlates +0.076 with hour-normalised engagement and 63% of candidates
    carry the single value 6.0; heat_score correlates -0.047 and was wired
    into nothing; the manual is_hot flag has never fired; the trending term
    is inert. What actually decided the batch was sha1(url) -- a fixed
    pseudo-random number, which is why 54% of eligible candidates over 72
    hours never entered a single ranking call and aged out unseen.

    The reason a chain of per-story calls cannot work here is visible in two
    independent measurements of the same shape. Asked to rate one story 1-10
    in isolation, the model answers 6.0 for 63% of them. Asked "which want
    does this satisfy", picking the best fit, it finds one for 94% of them.
    Asked the same question as a gate -- did this DELIVER an outcome, say 无
    when unsure -- only 5-9% pass, and those run at 1.34-1.41x normalised
    engagement against 0.99 for the rest, stable over three runs. A model
    rating things one at a time cannot discriminate; the same model comparing
    them can. So the judgement is made once, over the whole shortlist, at the
    moment the most context exists: what is in the pool, what we just
    published, which subjects are short, and what time it is.

    Fails open to None. Every caller falls back to the previous selection
    path, so a bad response, a timeout or a disabled flag reproduces exactly
    the old behaviour."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._client = create_openai_client(config)
        self._prompt = Path(config.editor.prompt_file).read_text(encoding="utf-8")

    def _payload(self, candidates: list[PublishCandidate], recent_titles: list[str],
                 subject_gaps: dict[str, float], trending: list[str], now) -> dict:
        ed = self._config.editor
        stories = []
        for i, c in enumerate(candidates, 1):
            stories.append({
                "id": str(i),
                "title": c.title[:180],
                "description": (c.description or "")[:ed.description_chars],
                "source": urlparse(c.url).netloc.replace("www.", "").lower() if c.url else "",
                "hours_old": round((now - c.published_at).total_seconds() / 3600, 1),
                "score": c.llm_score,
            })
        return {
            "today": now.strftime("%Y-%m-%d"),
            "trending_now": trending[:15],
            "recently_published": recent_titles[:ed.recent_titles_max],
            "subject_gaps": {k: round(v, 2) for k, v in subject_gaps.items() if abs(v) >= 0.05},
            "stories": stories,
        }

    async def _one_per_event(self, picks: list[EditorPick]) -> list[EditorPick]:
        """Keeps the highest-ranked pick of each underlying news event, using
        a second, deliberately narrow call.

        Two attempts to get this out of the main call both failed on the same
        pair. Told plainly that its picks must each be a different event, the
        model returned an incident report and a statement about that same
        incident at ranks 2 and 6. Asked to label each pick's underlying event
        so code could group them, it labelled the same pair "Terror suspects
        arrested in UK" and "Possible terror attack foiled" — two labels, one
        event. A rule buried in a long prompt is not where this gets decided.

        Asked on its own, over nothing but the chosen headlines, it is a
        comparison — which is the shape this model does well and the shape the
        whole of this layer is built on. One short call per cycle.

        Fails open: any error keeps every pick, which is the behaviour before
        this existed."""
        if len(picks) < 2:
            return picks
        listing = "\n".join(f"{i}. {p.candidate.title[:150]}" for i, p in enumerate(picks, 1))
        try:
            resp = await self._client.chat.completions.create(
                model=self._config.editor.model or self._config.openai.chat_model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SAME_EVENT_PROMPT},
                    {"role": "user", "content": listing},
                ],
            )
            groups = json.loads(resp.choices[0].message.content or "{}").get("groups") or []
        except Exception:
            logger.exception("EditorPicker: same-event grouping failed — keeping every pick")
            return picks

        drop: set[int] = set()
        for g in groups:
            nums = sorted({int(n) for n in g if str(n).isdigit() and 1 <= int(n) <= len(picks)})
            for n in nums[1:]:
                drop.add(n)
                logger.info("EditorPicker: dropped #%d %s — same event as #%d",
                            n, picks[n - 1].candidate.url, nums[0])
        return [p for i, p in enumerate(picks, 1) if i not in drop]

    def _enforce_caps(self, picks: list[EditorPick]) -> list[EditorPick]:
        """The caps are checkable, so they are checked here rather than
        trusted to the prompt. A pick that breaks one is dropped, not the
        whole response -- the rest of the ranking is still the editor's."""
        ed = self._config.editor
        subjects: Counter[str] = Counter()
        sources: Counter[str] = Counter()
        kept: list[EditorPick] = []
        for p in picks:
            src = urlparse(p.candidate.url).netloc.replace("www.", "").lower() if p.candidate.url else ""
            if p.subject and ed.per_batch_subject_cap > 0 and subjects[p.subject] >= ed.per_batch_subject_cap:
                logger.info("EditorPicker: dropped %s — subject cap for %s", p.candidate.url, p.subject)
                continue
            if src and ed.per_batch_source_cap > 0 and sources[src] >= ed.per_batch_source_cap:
                logger.info("EditorPicker: dropped %s — source cap for %s", p.candidate.url, src)
                continue
            subjects[p.subject or ""] += 1
            sources[src] += 1
            kept.append(p)
        return kept

    async def pick(self, candidates: list[PublishCandidate], recent_titles: list[str],
                   subject_gaps: dict[str, float], now,
                   trending: list[str] | None = None) -> list[EditorPick] | None:
        ed = self._config.editor
        if not ed.enabled or not candidates:
            return None
        # Short sequential ids, not page_ids. The first live run used 36-char
        # UUIDs over a 50-story list and the model returned valid ids attached
        # to reasons and labels belonging to OTHER stories -- it lost track of
        # which row it was writing about. A one- or two-digit handle is much
        # harder to drift on, and the echoed title below catches it when it
        # still does.
        by_id = {str(i): c for i, c in enumerate(candidates, 1)}
        system = self._prompt.replace("{pick_count}", str(ed.pick_count))
        try:
            resp = await self._client.chat.completions.create(
                model=ed.model or self._config.openai.chat_model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(
                        self._payload(candidates, recent_titles, subject_gaps, trending or [], now),
                        ensure_ascii=False)},
                ],
            )
            raw = json.loads(resp.choices[0].message.content or "{}").get("picks") or []
        except Exception:
            logger.exception("EditorPicker: call failed — falling back to the previous selection path")
            return None

        picks: list[EditorPick] = []
        seen: set[str] = set()
        for row in raw:
            if not isinstance(row, dict):
                continue
            c = by_id.get(str(row.get("id")))
            if c is None or c.page_id in seen:
                continue
            # The model echoes the title it believes it is choosing. When that
            # does not match the row its id points at, the pick is discarded:
            # the id and the judgement have come apart, so neither can be
            # trusted for this story. Compared on the first words only, since
            # the model may shorten a long headline.
            # Whitespace-normalised on both sides: a double space inside a
            # headline is not a mismatch, and one dropped a valid pick.
            echoed = " ".join(str(row.get("title") or "").lower().split())
            if echoed:
                head = " ".join(c.title.lower().split()[:5])
                if head and not (echoed.startswith(head[:28]) or head.startswith(echoed[:28])):
                    logger.warning(
                        "EditorPicker: dropped id=%s — echoed title %r does not match %r",
                        row.get("id"), echoed[:60], c.title[:60])
                    continue
            seen.add(c.page_id)
            raw_subject = row.get("subject")
            subject = raw_subject if raw_subject in TOPICS else None
            if subject is None and raw_subject:
                # Falls back to the catch-all rather than being left unset. The
                # prompt already says an unfitting story is 其他 and the model
                # still reached for a word of its own (it answered a payoff
                # label for a sports story), so code applies the rule the
                # prompt could not. Left unset the story would be invisible to
                # the mix controller; 其他 is what "none of these" means here.
                logger.warning("EditorPicker: subject %r is not in TOPICS — using 其他 for %s",
                               raw_subject, c.url)
                subject = "其他"
            raw_payoff = row.get("payoff")
            want = raw_payoff if raw_payoff in WANTS else None
            if want is None and raw_payoff:
                logger.warning("EditorPicker: payoff %r is not in WANTS — left unset for %s",
                               raw_payoff, c.url)
            picks.append(EditorPick(candidate=c, subject=subject, want=want,
                                    why=str(row.get("why") or "")[:300]))
        if not picks:
            logger.warning("EditorPicker: response held no usable picks — falling back")
            return None
        picks = self._enforce_caps(await self._one_per_event(picks))[:ed.pick_count]
        for i, p in enumerate(picks, 1):
            logger.info("EditorPicker: #%d %s [%s / %s] %s — %s",
                        i, p.candidate.url, p.subject, p.want, p.candidate.title[:70], p.why)
        return picks
