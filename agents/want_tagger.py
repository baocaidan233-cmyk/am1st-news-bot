from __future__ import annotations

import json
import logging

from core.config import AppConfig
from core.models import Candidate
from core.openai_client import create_openai_client

logger = logging.getLogger(__name__)

# The closed label set. A "want" is what the reader gets out of the story, not
# what the story is about -- the same subject can carry one or carry nothing.
# Measured 2026-09-26 on 571 of our own mature posts: a post satisfying any of
# these ran at a 1.30 normalised-engagement lift and broke out 3.4x as often as
# one satisfying none (p=0.0000 and p=0.0116), and the same split replicated in
# 10 of 10 peer channels over 1,815 posts, 9 of them individually significant.
# Nothing in the subject layer came close to that.
#
# The first twelve were distilled from the 45 best-performing posts of a peer
# channel; the last five were recovered by re-reading our own posts that the
# twelve-label version had called 无 while they performed well. That second
# pass is how a missing want shows up, and it should be repeated -- this list
# came from roughly 75 posts in total and is very unlikely to be complete.
WANTS = (
    "闸门关上了",
    "只有合法选民能投票",
    "他们在输入选民",
    "美国人的活被外国人抢走",
    "纳税人的钱在养敌人",
    "秩序回来了",
    "对方承认我们对",
    "媒体被打脸",
    "该付代价的终于付了",
    "美国重新被尊重",
    "生活成本/政府浪费",
    "伊斯兰威胁",
    "主权不受外人管",
    "保护孩子",
    "个人自由被夺回",
    "信仰与传统",
    "爱国自豪",
    "无",
)

_PROMPT = """This is a news item for a pro-Trump American news channel. Which UNDERLYING WANT
of that reader does it satisfy? Judge what the reader feels, not the subject. Pick the single
best fit.

闸门关上了 the flow is stopped: deportations carried out, caps cut, loopholes closed
只有合法选民能投票 only lawful voters vote: noncitizens purged, voter ID, proof of citizenship
他们在输入选民 the other side importing people to take power or change the electorate
美国人的活被外国人抢走 Americans losing work to foreign labour: H-1B, visas, offshoring
纳税人的钱在养敌人 our money funding what we oppose: NGOs, foreign aid, hostile groups
秩序回来了 order restored: agitators arrested, police backed, a crackdown that happened
对方承认我们对 an opponent, foreign leader, institution or defector concedes we were right
媒体被打脸 the press caught lying, humiliated, cut off, forced to retract
该付代价的终于付了 a named guilty party faces a real consequence: charged, fired, sentenced
美国重新被尊重 American strength recognised, allies falling in line, adversaries backing down
生活成本/政府浪费 the cost of ordinary life, or government squandering money
伊斯兰威胁 Islamist activity or extremism spreading inside America
主权不受外人管 America answering to no foreign body: the UN, ICC, WHO, treaties, globalists,
             foreign courts or foreign interference in American affairs
保护孩子 children shielded from harm: gender procedures on minors, sexual content in schools,
        abortion, predators, school safety, what kids are fed or shown
个人自由被夺回 a freedom restored or defended against the state: guns, speech, medical or
            vaccine mandates, religious practice, bodily autonomy, surveillance
信仰与传统 Christianity and traditional family life affirmed in public life
爱国自豪 plain pride in America, its military, its symbols, its history
无 none of these fits

Reply with EXACTLY this JSON and no other keys, the value copied exactly from the list above:
{"w": "闸门关上了"}"""


class WantTagger:
    """One gpt-4o-mini call per candidate that reaches the pool, on title +
    description, assigning the reader want the story satisfies.

    Deliberately its own call rather than a second field on the topic tagger's,
    which is the obvious saving and was measured on 2026-09-26: folding the two
    together left the subject field intact (88% agreement against a 91%
    self-jitter baseline) but cut the want field's has/has-not agreement from
    93% to 77%. The model does the first task well and the second carelessly.

    Also deliberately an LLM call rather than a comparison against embeddings
    we already compute for dedup, which would have been free. Tested the same
    day: best agreement with the LLM labels was 72%, and using the embedding
    judgement to predict engagement collapsed the effect from 1.30x to 1.07x,
    with the breakout split running backwards at the looser threshold. The
    reason is visible once stated -- "deportations carried out" and
    "deportation policy debated" sit next to each other in embedding space and
    on opposite sides of this question. A want is about whether the thing
    happened and to whom, which is exactly what a similarity score cannot see.

    Fails open to None, like every other enrichment here: an untagged candidate
    competes exactly as it did before this existed."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._client = create_openai_client(config)

    async def tag(self, item: Candidate) -> str | None:
        text = f"{item.title}\n{item.description}".strip()[:600]
        if not text:
            return None
        try:
            resp = await self._client.chat.completions.create(
                model=self._config.openai.chat_model,
                temperature=0,
                max_tokens=24,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _PROMPT},
                    {"role": "user", "content": text},
                ],
            )
            label = json.loads(resp.choices[0].message.content or "{}").get("w")
        except Exception:
            logger.exception("WantTagger: tagging failed for %s — continuing untagged", item.url)
            return None

        if label not in WANTS:
            logger.warning("WantTagger: model returned unknown label %r for %s — treating as untagged", label, item.url)
            return None
        return label
