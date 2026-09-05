from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from core.config import AppConfig
from core.openai_client import create_openai_client


class StalenessChecker:
    """Separate, single-purpose LLM call (2026-09-05) — decides whether an
    article is fundamentally stale analysis/retrospective about an event
    that already happened, vs. genuinely fresh news, BEFORE Writer ever
    runs on it. Real incident: two op-eds analyzing deals that were 9 days
    and 2 weeks old (a Venezuela oil deal, an Ethiopia defense agreement)
    both got written up as if breaking news.

    Deliberately NOT folded into content_gen_prompt.txt's own instructions
    — three separate attempts to make Writer self-police this in the same
    call (a worked example, an explicit today's-date anchor, a priority
    note telling SUBSTANCE REQUIREMENT to defer to it) all failed on the
    same two real test articles. SUBSTANCE REQUIREMENT's own three-fold
    reinforcement (imperative language + worked examples + its own
    self-check demanding "a specific fact, number, quote, or date") kept
    winning regardless of how the staleness rule was phrased or where it
    was placed in the prompt. This matches this codebase's own established
    precedent: EventVerifier.classify_subtype() is a deliberately separate
    call from same_event(), after an 2026-08-09 ablation test found asking
    both in one prompt biases the model toward one answer ~30% of the
    time (see EntityVerifierConfig's docstring) — a single call juggling
    multiple judgments is less reliable than separate, single-purpose
    calls, not just here.

    Runs on config.openai.chat_model, same as every other judgment call in
    this codebase (see OpenAIConfig's docstring on why nano models were
    reverted everywhere 2026-09-01)."""

    def __init__(self, config: AppConfig) -> None:
        self._client = create_openai_client(config)
        self._model = config.openai.chat_model
        self._prompt = Path(config.openai.staleness_check_prompt_file).read_text(encoding="utf-8")

    async def is_stale(self, title: str, article: str) -> tuple[bool, str]:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        user_message = self._prompt.format(today=today, title=title, article=article[:6000])
        kwargs = dict(model=self._model, messages=[{"role": "user", "content": user_message}])
        if self._model.startswith("gpt-5"):
            kwargs["max_completion_tokens"] = 80
            kwargs["reasoning_effort"] = "minimal"
        else:
            kwargs["temperature"] = 0
            kwargs["max_tokens"] = 80
        resp = await self._client.chat.completions.create(**kwargs)
        raw = (resp.choices[0].message.content or "").strip()
        verdict = ""
        for line in raw.splitlines():
            if line.upper().startswith("VERDICT"):
                verdict = line.split(":", 1)[1].strip().upper()
                break
        return verdict.startswith("STALE"), raw
