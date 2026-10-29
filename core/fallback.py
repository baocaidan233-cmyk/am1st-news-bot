"""Request parameters for openai.fallback_model (2026-10-29, user: 所有回退全部改成 luna).

gpt-6 and gpt-5 reject `temperature` (400) and count reasoning in
max_completion_tokens, so the reasoning room is added on top of the answer's
own budget; older models keep temperature and max_tokens."""
from __future__ import annotations

_REASONING_ROOM = 3000


def fallback_params(config, max_tokens: int | None = None, temperature: float = 0) -> dict:
    model = config.openai.fallback_model
    if model.startswith(("gpt-5", "gpt-6")):
        d = {"model": model, "reasoning_effort": config.openai.fallback_reasoning_effort}
        if max_tokens:
            d["max_completion_tokens"] = max_tokens + _REASONING_ROOM
        return d
    d = {"model": model, "temperature": temperature}
    if max_tokens:
        d["max_tokens"] = max_tokens
    return d
