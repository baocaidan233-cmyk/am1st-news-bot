from __future__ import annotations

import json
import logging

from core.config import AppConfig
from core.models import Candidate
from core.openai_client import create_openai_client

logger = logging.getLogger(__name__)

# The closed label set. These are exactly the buckets the 2026-09-25 audit
# measured real Gettr engagement on (534 of our own mature posts, plus 2,293
# posts across 13 peer channels) — see the memory note project_am1st_payoff_mix.
# config.topic_mix.targets is keyed on these same strings, so renaming, adding
# or splitting a label here silently orphans a target and invalidates the
# measured mix it was derived from. Change both together or neither.
#
# Deliberately NOT the scoring prompt's 21 Core Themes: those exist to decide
# whether a story is on-theme at all (a gate), and several of them overlap
# heavily (Trump Agenda / DOJ Lawfare / Government Corruption all cover the
# same lawfare stories). These buckets are chosen to be mutually exclusive and
# to carry measurably different engagement, which is a different job.
TOPICS = (
    "选举诚信",
    "移民边境ICE",
    "跨性别与儿童",
    "新冠追责",
    "Antifa左翼暴力",
    "媒体与审查",
    "司法武器化",
    "犯罪治安",
    "经济通胀关税",
    "外交与战争",
    "中国CCP",
    "以色列中东",
    "教育与学校",
    "枪权",
    "堕胎",
    "共和党内部与人物",
    "其他",
)

_PROMPT = """Assign this news headline to exactly ONE subject bucket. Pick the subject, not the tone.
选举诚信 elections, voter rolls, ballots, redistricting, election law/fraud, voter ID
移民边境ICE immigration, border, ICE, deportation, illegal aliens, asylum, birthright citizenship
跨性别与儿童 transgender, gender care for minors, girls' sports, drag, child sexual content
新冠追责 COVID, lockdowns, masks, vaccines, mandates, Fauci, virus origin
Antifa左翼暴力 Antifa, riots, left political violence, attacks on conservatives
媒体与审查 press, CNN/MSNBC, censorship, deplatforming, Big Tech, press access
司法武器化 DOJ/FBI targeting conservatives, J6, lawfare, special counsel, Biden family probes
犯罪治安 ordinary crime, police, soft-on-crime prosecutors, drugs, cartels, trafficking
经济通胀关税 economy, inflation, prices, tariffs, jobs, markets, energy
外交与战争 foreign policy, wars, NATO, Ukraine, Iran, the UN, treaties, troop deployments
中国CCP China, the CCP, Xi, Chinese influence or espionage
以色列中东 Israel, Gaza, Hamas, antisemitism, the Middle East
教育与学校 schools, curricula, universities, teachers, school boards, DEI in education
枪权 guns
堕胎 abortion
共和党内部与人物 GOP internal politics, personalities, endorsements, campaigns, ceremonies, appointments
其他 none of the above

Reply with EXACTLY this JSON and no other keys, and the value must be one of the
bucket names above, copied exactly: {"t": "选举诚信"}"""


class TopicTagger:
    """One cheap gpt-4o-mini call per candidate that survives scoring, on
    title + description only.

    Deliberately a SEPARATE call rather than an extra field on
    prompts/scoring_prompt.txt. Adding a field to an already-large prompt has
    a measured history in these bots of degrading the fields that were
    already there (the DailyNews scope-gate work has the clean before/after),
    and scoring_prompt.txt is the one prompt whose output gates the entire
    pool. The marginal cost of keeping them apart is ~530 short calls a day.

    Fails open: any error, malformed JSON, or a label outside TOPICS returns
    None, and every consumer treats None as "no topic adjustment" — an
    untagged candidate competes exactly as it does today. This is the same
    convention as every other best-effort enrichment in this codebase."""

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
            label = json.loads(resp.choices[0].message.content or "{}").get("t")
        except Exception:
            logger.exception("TopicTagger: tagging failed for %s — continuing untagged", item.url)
            return None

        if label not in TOPICS:
            logger.warning("TopicTagger: model returned unknown label %r for %s — treating as untagged", label, item.url)
            return None
        return label
