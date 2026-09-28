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
                 subject_gaps: dict[str, float], now) -> dict:
        ed = self._config.editor
        stories = []
        for c in candidates:
            stories.append({
                "id": c.page_id,
                "title": c.title[:180],
                "description": (c.description or "")[:ed.description_chars],
                "source": urlparse(c.url).netloc.replace("www.", "").lower() if c.url else "",
                "hours_old": round((now - c.published_at).total_seconds() / 3600, 1),
                "score": c.llm_score,
            })
        return {
            "today": now.strftime("%Y-%m-%d"),
            "recently_published": recent_titles[:ed.recent_titles_max],
            "subject_gaps": {k: round(v, 2) for k, v in subject_gaps.items() if abs(v) >= 0.05},
            "stories": stories,
        }

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
                   subject_gaps: dict[str, float], now) -> list[EditorPick] | None:
        ed = self._config.editor
        if not ed.enabled or not candidates:
            return None
        by_id = {c.page_id: c for c in candidates}
        system = self._prompt.replace("{pick_count}", str(ed.pick_count))
        try:
            resp = await self._client.chat.completions.create(
                model=ed.model or self._config.openai.chat_model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(
                        self._payload(candidates, recent_titles, subject_gaps, now), ensure_ascii=False)},
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
            c = by_id.get(row.get("id"))
            if c is None or c.page_id in seen:
                continue
            seen.add(c.page_id)
            subject = row.get("subject") if row.get("subject") in TOPICS else None
            want = row.get("want") if row.get("want") in WANTS else None
            picks.append(EditorPick(candidate=c, subject=subject, want=want,
                                    why=str(row.get("why") or "")[:300]))
        if not picks:
            logger.warning("EditorPicker: response held no usable picks — falling back")
            return None
        picks = self._enforce_caps(picks)[:ed.pick_count]
        for i, p in enumerate(picks, 1):
            logger.info("EditorPicker: #%d %s [%s / %s] %s — %s",
                        i, p.candidate.url, p.subject, p.want, p.candidate.title[:70], p.why)
        return picks
