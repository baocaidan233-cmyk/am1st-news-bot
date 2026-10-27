"""Election-fraud stories first (2026-10-27). User: 大家在乎的是选举舞弊，尤其左派偷盗选举 -- a 7 on
election fraud should go out ahead of an ordinary 8. A candidate counts when its title names fraud or a
stolen / rigged / contested vote and is about an election. "Ballot" alone is not
enough ("the Second Amendment is on the ballot" is a ballot measure) (topic 选举诚信, or an election word in the title).
The topic alone is too wide: replayed on 10-24..10-27, boosting all of 选举诚信 moved "Trump calls El-Sayed
'jihadist'" and a poll of Jewish voters to the top. Titles only, no model call."""
from __future__ import annotations

import re

_FRAUD = re.compile(
    r"(?i)\b(fraud\w*|rigg\w*|stole[n]?|steal\w*|cheat\w*|non-?citizens?|illegal(ly)? vot\w*|"
    r"(migrants?|aliens?|illegals) (cast|vot\w*)|ballots|machines?|vote-counting|"
    r"voter rolls?|vote counts?|recounts?|certif\w*|annul\w*|invalid|irregularit\w*|"
    r"dark money|shenanigans|double vot\w*|dead voters?|harvest\w*|mail-in|absentee|voter id|no id|"
    r"election (monitors?|integrity|interference)|contest\w* (the )?(election|results?|vote))\b")
_ELECTION = re.compile(r"(?i)\b(elect\w*|vot\w*|ballots?|polls? (site|place|station)s?|polling|midterms?)\b")


def is_election_fraud(title: str, topic: str | None) -> bool:
    t = title or ""
    return bool(_FRAUD.search(t)) and (topic == "选举诚信" or bool(_ELECTION.search(t)))
