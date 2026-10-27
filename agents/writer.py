from __future__ import annotations

import re
from pathlib import Path

from core.config import AppConfig
from core.openai_client import create_openai_client
from core.date_context import date_block
from core.caption_meta import drop_meta_clauses

NO_COMMENT = "No comment"

# Strips markdown emphasis (*_`#) and trailing punctuation before comparing —
# a real published post (2026-08-06) revealed the model sometimes wraps its
# "decline to write this" signal in markdown ("**No comment**"), which an
# exact-string match doesn't recognize as the same thing. That post then
# proceeded through ranking/dedup/publish as if it were real content — see
# project_am1st_migration memory's 2026-08-06 note for the full incident.
_MARKDOWN_RE = re.compile(r"[*_`#]+")


class Writer:
    """Content generation — same prompt as the original n8n workflow's 'Gen
    Gettr Post' node, ported verbatim (prompts/content_gen_prompt.txt).
    Deliberately a separate LLM call from Scorer, not merged into one
    request — Scoring decides "is this worth reporting", this decides "how
    to write it"; keeping them separate matches the original's own division
    of labor (see am1st_pipeline.html #p5).

    Called from the publish cycle only (2026-08-05 — moved out of
    ingestion, same reasoning as agents/extractor.py's docstring): takes
    plain title/article text rather than a specific Candidate type so it
    works for whichever model the caller has on hand."""

    def __init__(self, config: AppConfig) -> None:
        self._client = create_openai_client(config)
        self._model = config.openai.writer_model or config.openai.chat_model
        # gpt-5 and gpt-6 families only (2026-10-11, gpt-6 2026-10-27): reasoning effort, and no temperature
        # (none is sent today either way).
        self._extra = ({"reasoning_effort": config.openai.writer_reasoning_effort}
                       if self._model.startswith(("gpt-5", "gpt-6")) and config.openai.writer_reasoning_effort else {})
        self._system_prompt = Path(config.openai.content_gen_prompt_file).read_text(encoding="utf-8")

    async def write(self, title: str, article: str, context: str = "", is_opinion: bool = False, published_at=None,
                    english_retry: bool = False, new_info: str = "") -> str:
        """`context` (2026-08-31) — optional prior-developments/related-
        events summary for this story, built by main_publish.py from
        core/qdrant_store.py's EventStore (timeline + related_event_ids on
        the matched event, if any). Kept as its own labeled section in the
        USER message, separate from the static system prompt file (see
        prompts/content_gen_prompt.txt's "OPTIONAL BACKGROUND" section for
        the model-facing instructions on how to use it) — appended only
        when non-empty, so omitting it reproduces today's exact behavior.

        `is_opinion` (2026-09-05) — set when agents/staleness_checker.py
        classified this article as OPINION: a genuine analysis/commentary
        piece about an older event with real argument/expert input, not a
        pure rehash (that gets dropped before write() is ever called) and
        not fresh news (is_opinion stays False). Appends a short framing
        instruction rather than asking Writer to detect this itself —
        three earlier attempts at self-detection inside this same call all
        failed (see StalenessChecker's docstring); telling Writer HOW to
        frame something it's already been told IS opinion is a much
        simpler ask than asking it to independently classify freshness
        while also fighting SUBSTANCE REQUIREMENT's pull toward citing
        dates as proof of freshness.

        Today's date — kept in the user message even though the primary
        staleness gate now happens before write() is ever called (see
        StalenessChecker); may still help the existing "BREAKING: only for
        fresh events" judgment in OUTPUT FORMAT.
        `published_at` (2026-09-28) — the article's own publication date, used
        by core/date_context.py to resolve the source's relative dates before
        the model sees them. Optional and fails open: without it the message
        carries today's date alone, which is what it carried before.
        `new_info` (2026-10-22, AI-24): for a follow-up to a story this channel
        already posted, what the ingestion judge found new for our readers.
        The post leads with it; empty for everything else.
        """
        user_message = self._user_message(title, article, context, is_opinion, published_at, english_retry, new_info)
        resp = await self._client.chat.completions.create(
            model=self._model,
            **self._extra,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return drop_meta_clauses((resp.choices[0].message.content or "").strip()) or NO_COMMENT

    async def rewrite(self, title: str, article: str, draft: str, problems: list[str], context: str = "",
                      is_opinion: bool = False, published_at=None) -> str:
        """One rewrite of a caption the fact check failed (2026-10-13, Linear
        AI-74, core/fact_check.py; port of China Breaks 5ea76b1): the same
        request as write(), the failed draft as the model's own answer, then
        the checker's reasons. Offline on China Breaks, 12 of 13 wrong drafts
        came back right this way. The caller checks the result again."""
        user_message = self._user_message(title, article, context, is_opinion, published_at, False)
        fix = ("A fact-checker compared your post with the article and found these problems:\n"
               + "\n".join(f"- {p}" for p in problems)
               + "\n\nRewrite the post so that every fact matches the article. Fix these problems and "
                 "nothing else; keep the rest, the format and the word limit. Output only the post.")
        resp = await self._client.chat.completions.create(
            model=self._model,
            **self._extra,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": draft},
                {"role": "user", "content": fix},
            ],
        )
        return drop_meta_clauses((resp.choices[0].message.content or "").strip()) or NO_COMMENT

    @staticmethod
    def _user_message(title, article, context, is_opinion, published_at, english_retry, new_info="") -> str:
        user_message = (f"{date_block(article, published_at)}\n\n"
                        f"Title:  {title}\n\nArticle: {article}")
        if context:
            user_message += f"\n\nBackground: {context}"
        if new_info:
            user_message += (
                "\n\nFollow-up: this channel has already posted this story, so our readers know the earlier "
                f"part. What this article adds for them: {new_info}\n"
                "Open the post with that new development, written as news in its own right: do not compare it "
                "with earlier reports or say what is new about it. Mention what they already know only as "
                "brief context, after it, if at all."
            )
        if is_opinion:
            user_message += (
                "\n\nNote: this article is analysis/commentary about an event "
                "that already happened, not a fresh news report. Frame the "
                "post accordingly — as opinion/analysis (e.g. \"Analysts "
                "argue...\", \"The debate centers on...\", naming the actual "
                "expert or source where the article does) — do not write it "
                "as if the underlying event just happened today, and do not "
                "use \"BREAKING.\""
            )
        if english_retry:
            # AI-73: set by main_publish.py when the first caption carried a
            # letter outside the Latin script. One retry with the rule
            # restated where the model looks last; a second such caption is
            # dropped, never published. Same as Leading News.
            user_message += (
                "\n\nIMPORTANT: your post must be written entirely in English, "
                "whatever language the article is in. Translate every name and word; "
                "do not copy any of the original script."
            )
        return user_message

    @staticmethod
    def is_no_comment(text: str) -> bool:
        cleaned = _MARKDOWN_RE.sub("", text).strip().rstrip(".!").strip().lower()
        return cleaned == NO_COMMENT.lower()
