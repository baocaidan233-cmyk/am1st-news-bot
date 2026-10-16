"""AI-89 (2026-10-16): the cache marker for fact-checked captions, and the
fact check's gpt-4o-mini fallback when Haiku is broken. Pure code: fake
redis and fake model clients, no network, no keys.

Run: python3 -m pytest tests/test_caption_fact_check.py, or python3 tests/test_caption_fact_check.py."""
import asyncio
import json
import os
import sys
import types
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# core/fact_check.py imports anthropic at the top; the fakes below never use it.
try:
    import anthropic  # noqa: F401
except ImportError:
    sys.modules["anthropic"] = types.ModuleType("anthropic")

from core.config import AppConfig
from core.fact_check import FactChecker, _parse
from core.redis_store import CaptionCache


class FakeRedis:
    def __init__(self, broken=False):
        self.d, self.broken = {}, broken

    async def get(self, k):
        if self.broken:
            raise ConnectionError("down")
        return self.d.get(k)

    async def set(self, k, v, ex=None):
        if self.broken:
            raise ConnectionError("down")
        self.d[k] = v

    async def delete(self, *ks):
        if self.broken:
            raise ConnectionError("down")
        for k in ks:
            self.d.pop(k, None)


def _cache(broken=False):
    cc = CaptionCache(AppConfig())  # no REDIS_URL in a bare AppConfig: client None
    cc._client = FakeRedis(broken)
    return cc


GOOD = json.dumps({"claims": [{"text": "a", "label": "SUPPORTED", "reason": "ok"},
                              {"text": "b", "label": "NOT_FOUND", "reason": "not in article"}],
                   "verdict": "FAIL"})


def test_parse():
    verdict, claims, problems = _parse(GOOD)
    assert verdict == "FAIL" and len(claims) == 2 and problems == ['"b": not in article']
    for bad in ("not json", json.dumps({"verdict": "MAYBE", "claims": []}), json.dumps({"claims": []})):
        try:
            _parse(bad)
        except Exception:
            continue
        raise AssertionError(f"accepted {bad!r}")


class FakeHaiku:
    def __init__(self, text=None, exc=None, refusal=False):
        self.text, self.exc, self.refusal, self.calls = text, exc, refusal, 0
        self.messages = self

    async def create(self, **kw):
        self.calls += 1
        if self.exc:
            raise self.exc
        return SimpleNamespace(stop_reason="refusal" if self.refusal else "end_turn",
                               usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                               content=[SimpleNamespace(type="text", text=self.text or "")])


class FakeOpenAI:
    def __init__(self, text=None, exc=None):
        self.text, self.exc, self.kwargs = text, exc, []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kw):
        self.kwargs.append(kw)
        if self.exc:
            raise self.exc
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.text, refusal=None))],
                               usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))


def _checker(haiku, openai):
    fc = FactChecker.__new__(FactChecker)
    fc.enabled, fc._model, fc._effort = True, "claude-haiku-5-5", "low"
    fc._system, fc._fix_system = "SYS", "FIXSYS"
    fc._client, fc._openai, fc._openai_model, fc._timeout = haiku, openai, "gpt-4o-mini", 60
    fc._log = lambda *a, **k: None
    return fc


PASS = json.dumps({"claims": [], "verdict": "PASS"})


def test_check_falls_back_only_when_haiku_is_broken():
    async def run():
        # Haiku answers: no OpenAI call.
        oa = FakeOpenAI(PASS)
        assert (await _checker(FakeHaiku(GOOD), oa).check("t", "a", "p"))["verdict"] == "FAIL"
        assert not oa.kwargs
        # Raises / refuses / unreadable: gpt-4o-mini answers with strict json_schema.
        for h in (FakeHaiku(exc=TimeoutError()), FakeHaiku(PASS, refusal=True), FakeHaiku("{oops")):
            oa = FakeOpenAI(GOOD)
            r = await _checker(h, oa).check("t", "a", "p")
            assert r["verdict"] == "FAIL" and r["problems"] == ['"b": not in article'], r
            kw = oa.kwargs[0]
            assert kw["model"] == "gpt-4o-mini" and kw["messages"][0] == {"role": "system", "content": "SYS"}
            assert kw["response_format"]["json_schema"]["strict"] is True
        # Both broken, or no fallback client: ERROR (the caller lets it through).
        assert (await _checker(FakeHaiku(exc=RuntimeError()), FakeOpenAI(exc=RuntimeError())).check("t", "a", "p"))["verdict"] == "ERROR"
        assert (await _checker(FakeHaiku(exc=RuntimeError()), FakeOpenAI("nope")).check("t", "a", "p"))["verdict"] == "ERROR"
        assert (await _checker(FakeHaiku(exc=RuntimeError()), None).check("t", "a", "p"))["verdict"] == "ERROR"
    asyncio.run(run())


def test_fix_falls_back_only_when_haiku_is_broken():
    async def run():
        post = "Trump met Xi on Monday."
        # Haiku broken -> 4o-mini's correction, same post-processing.
        oa = FakeOpenAI("Trump met Xi  on Tuesday.")
        assert await _checker(FakeHaiku(exc=TimeoutError()), oa).fix("t", "a", post, ["x"]) == "Trump met Xi on Tuesday."
        assert oa.kwargs[0]["messages"][0]["content"] == "FIXSYS" and "response_format" not in oa.kwargs[0]
        oa = FakeOpenAI("Trump met Xi on Tuesday.")
        assert await _checker(FakeHaiku(refusal=True), oa).fix("t", "a", post, ["x"]) == "Trump met Xi on Tuesday."
        # A usable-but-rejected Haiku answer is not an error: no fallback.
        for text in (post, "", "The article says Trump met Xi."):
            oa = FakeOpenAI("Trump met Xi on Tuesday.")
            assert await _checker(FakeHaiku(text), oa).fix("t", "a", post, ["x"]) is None
            assert not oa.kwargs
        # The fallback's answer goes through the same unusable-output checks.
        assert await _checker(FakeHaiku(exc=TimeoutError()), FakeOpenAI(post)).fix("t", "a", post, ["x"]) is None
        # Both broken: None (the caller drops).
        assert await _checker(FakeHaiku(exc=TimeoutError()), FakeOpenAI(exc=RuntimeError())).fix("t", "a", post, ["x"]) is None
        assert await _checker(FakeHaiku(exc=TimeoutError()), None).fix("t", "a", post, ["x"]) is None
    asyncio.run(run())


# ---- CaptionCache marker: stored in the caption's own value (10-16) --------
# One key per caption, so a cache hit stays one Redis command (Upstash bills
# per command); is_checked() after get() needs no call of its own.

class _MarkRedis:
    def __init__(self, fail=False):
        self.data, self.fail, self.calls = {}, fail, 0

    async def get(self, key):
        self.calls += 1
        if self.fail:
            raise ConnectionError("down")
        return self.data.get(key)

    async def set(self, key, value, ex=None, nx=False):
        self.calls += 1
        if self.fail:
            raise ConnectionError("down")
        self.data[key] = value

    async def delete(self, *keys):
        self.calls += 1
        if self.fail:
            raise ConnectionError("down")
        for k in keys:
            self.data.pop(k, None)


def _mark_cache(client):
    c = CaptionCache.__new__(CaptionCache)
    c._prefix, c._ttl, c._client, c._checked = "test:caption:", 172800, client, {}
    return c


def test_marker_rides_in_the_caption_value():
    async def run():
        r = _MarkRedis()
        await _mark_cache(r).set("h1", "Beijing said X.")
        assert r.data == {"test:caption:h1": "Beijing said X."}          # unchecked: stored as before
        c = _mark_cache(r)
        assert await c.get("h1") == "Beijing said X."
        assert not await c.is_checked("h1", "Beijing said X.")           # cached before the marker existed
        await c.set("h1", "Beijing said X.", checked=True)
        assert list(r.data) == ["test:caption:h1"]                       # no second key
        c = _mark_cache(r)
        assert await c.get("h1") == "Beijing said X."                    # the marker never leaks into the text
        n = r.calls
        assert await c.is_checked("h1", "Beijing said X.")
        assert r.calls == n                                              # answered without a Redis call
        assert not await c.is_checked("h1", "Beijing said Y.")           # text changed after its check
        await c.mark_checked("h1", "Beijing said Z.")
        assert await _mark_cache(r).is_checked("h1", "Beijing said Z.")  # without a get() first: one GET
        await c.delete("h1")
        assert r.data == {} and await c.get("h1") is None
        assert not await c.is_checked("h1", "Beijing said Z.")
    asyncio.run(run())


def test_marker_fails_open():
    async def run():
        down = _mark_cache(_MarkRedis(fail=True))
        assert await down.get("h1") is None
        assert not await down.is_checked("h1", "x")
        await down.set("h1", "x", checked=True)   # no raise
        await down.mark_checked("h1", "x")
        await down.delete("h1")
        assert not await down.is_checked("h1", "x")
        none = _mark_cache(None)
        assert not await none.is_checked("h1", "x")
        await none.mark_checked("h1", "x")
        await none.delete("h1")
    asyncio.run(run())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
