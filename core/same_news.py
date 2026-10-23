"""Would this post repeat news the channel already posted? The publish-side
judge (agents/posted_dedup_checker.py). Same file in AM1ST, China Breaks,
Leading News and Market Watcher; each channel's own note is in its
prompts/same_news_prompt.txt.

Caidan, 2026-10-12 (Linear AI-60). Each side goes to the judge as our post plus
its source's headline and first paragraph, in one call. A Gettr scan that day
found 93 repeats in about 3.5 days across the six accounts, 83 of them pairs
the old judge had been asked about and called different -- mostly over figures
and details its prompt told it to treat as a different event. Replayed on 562
blind-labelled pairs the old judge had all let through (counted per post):

  old prompt, source headline + first paragraph   48/93 repeats, 25/155 good posts blocked
  this prompt, source only                        21/93, 11/155
  this prompt, our post + source                  35/93,  5/155
  this prompt, our post only                      31/93, 12/155

The source alone misses the repeats our writer makes: two different source
stories (a phone call, an analyst's commentary on it) both written up as the
same deal. The post alone has nothing to check the writer against.

An UNCERTAIN verdict counts as a duplicate: the candidate waits and a
different one is published, and two such verdicts retire it like any other
duplicate. A reply that cannot be read raises, which the caller treats as a
failed check, not a verdict."""
from __future__ import annotations

import json
from pathlib import Path

from core import haiku
from core.config import AppConfig
from core.event_identity import event_identity_text
from core.openai_client import create_openai_client

# A first "paragraph" shorter than this is a byline, a dateline, a photo
# credit or an "Advertisement" line, not the lead.
_LEAD_MIN_CHARS = 80
_LEAD_MAX_CHARS = 600
_POST_MAX_CHARS = 1200

_SCHEMA = {"type": "json_schema", "json_schema": {"name": "same_news", "strict": True, "schema": {
    "type": "object", "additionalProperties": False,
    "required": ["published", "candidate", "verdict", "reason"],
    "properties": {
        "published": {"type": "string"},
        "candidate": {"type": "string"},
        "verdict": {"type": "string", "enum": ["SAME", "DIFFERENT", "UNCERTAIN"]},
        "reason": {"type": "string"},
    }}}}


def lead_paragraph(body: str) -> str:
    """The article's first real paragraph, from the extracted full text.
    "" if the body has none."""
    for para in (body or "").split("\n"):
        p = " ".join(para.split())
        if len(p) >= _LEAD_MIN_CHARS:
            return p[:_LEAD_MAX_CHARS]
    return ""


def news_record(post: str, title: str, lead: str, description: str = "") -> str:
    """One side as the judge reads it. A post written before 2026-10-12 has no
    stored paragraph and shows the first sentence of its RSS description."""
    lines = [f"Our post: {(post or '').strip()[:_POST_MAX_CHARS]}"]
    if title:
        lines.append(f"Source headline: {title.strip()}")
    if lead:
        lines.append(f"Source first paragraph: {lead}")
    elif description:
        summary = event_identity_text("", description).strip()
        if summary:
            lines.append(f"Source summary: {summary}")
    return "\n".join(lines)


class SameNewsJudge:
    def __init__(self, config: AppConfig) -> None:
        self._client = create_openai_client(config)
        self._model = config.openai.chat_model
        self._prompt = Path(config.entity_verifier.same_news_prompt_file).read_text(encoding="utf-8")
        # 2026-10-15 (AI-60, user: switch every judge Haiku measured better on):
        # Claude Haiku first, self._model only when the Haiku key is out of credit (core/haiku.py).
        # On the 10-14 Gettr pairs Haiku stopped 85% of China Breaks' repeats
        # (4o-mini 20%), 53-72% on Leading News (8-14%), 45% on Market Watcher
        # (0%), blocking 0-5% of different news.
        ev = config.entity_verifier
        self._haiku_key = (getattr(ev, "haiku_api_key", "") or getattr(ev, "guard_judge_api_key", "")) if getattr(ev, "same_news_haiku", True) else ""

    async def judge(self, published: str, candidate: str) -> tuple[str, str]:
        """("SAME" | "DIFFERENT" | "UNCERTAIN", raw reply)."""
        user = f"PUBLISHED:\n{published}\n\nCANDIDATE:\n{candidate}"
        if self._haiku_key:
            # The prompt is ~800 tokens and a candidate is compared with up to 5 posts in a row: cached.
            d = await haiku.ask(self._haiku_key, self._prompt, user, _SCHEMA["json_schema"]["schema"], cache=True, cache_ttl="1h")
            if d and d.get("verdict") in ("SAME", "DIFFERENT", "UNCERTAIN"):
                return d["verdict"], json.dumps({**d, "model": haiku.MODEL}, ensure_ascii=False)
            # User 10-16 (AI-69): Haiku broken or out of credit, the same
            # question goes to OpenAI below (was: only out of credit, 10-15).
        kwargs = dict(model=self._model, response_format=_SCHEMA, messages=[
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": user},
        ])
        if self._model.startswith("gpt-5"):
            kwargs["max_completion_tokens"] = 400
            kwargs["reasoning_effort"] = "minimal"
        else:
            kwargs["temperature"] = 0
            kwargs["max_tokens"] = 250
        resp = await self._client.chat.completions.create(**kwargs)
        raw = (resp.choices[0].message.content or "").strip()
        verdict = json.loads(raw)["verdict"]
        if verdict not in ("SAME", "DIFFERENT", "UNCERTAIN"):
            raise ValueError(f"same_news: unexpected verdict {verdict!r}")
        return verdict, raw


_judge: SameNewsJudge | None = None


def get_judge(config: AppConfig) -> SameNewsJudge:
    global _judge
    if _judge is None:
        _judge = SameNewsJudge(config)
    return _judge
