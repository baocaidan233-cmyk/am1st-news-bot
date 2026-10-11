"""Train the vector stance check -- see core/stance_guard.StanceVectorGuard.

What it learns is whose side a headline is written from, not whether it is
negative. The outlet is never an input to the model:

  1  a left-hostile outlet's headline about Trump's side (Daily Beast, HuffPost,
     MS NOW, the Guardian, Rolling Stone, The New Republic ...) that attacks in
     its own voice -- hand-labelled, see LABELS below
  0  the same outlets' headlines that report someone else's attack, or just
     report (hand-labelled); a right-leaning outlet's headline about the same
     side, a wire service's, and every title this channel has published that
     did not come from one of the hostile outlets above

A titles-only logistic regression on text-embedding-3-small, the same embedding
model the pipeline already uses.

The first version (2026-10-07) was trained on everything logged before
2026-10-03 00:00 UTC and measured on what came after. At the deployed 0.8
threshold: 49% of hostile-outlet headlines in the test period flagged (the word
list: 12.9%); 3.6% of right-outlet and 5.6% of wire headlines; and of the 210
posts the channel actually published after the cut, 4 -- all four of them
hostile-frame posts. Hand-flagged errors are never training data: titles listed
in HOLDOUT stay out so they remain an honest check.

2026-10-08: labelling every hostile-outlet headline 1 taught the model the
outlet, not the voice -- of 19 candidates it blocked in a day, about 12 only
reported an attack (Warnock, Carville, "Trump rebukes advisers", "heckled at
rally"). Owner's rule: our posts may not attack Trump or use attack words, but
may report how someone attacks him. So the 494 hostile-outlet headlines logged
up to then were labelled by hand (by Claude, not the owner) in LABELS: 1 =
attacks in its own voice, 0 = reports. Hostile-outlet headlines not in LABELS are
left out. Same cut, same negatives, measured after the cut at 0.8:
                                        distant  hand
  hostile outlet, labelled attack        63.2%  59.7%   (should block)
  hostile outlet, labelled report        24.7%   7.1%   (should pass)
  right outlets / wires                  3.4% / 10.6%   0.9% / 0%
  our own posts, 490 MAGA-account posts  0% / 0.2%      0% / 0%

2026-10-11 (AI-50): after the ProPublica "DHS Intimidates a Border Tribe" post
(scored 0.19), 2,271 more headlines were hand-labelled (by five Claude
subagents on one written rubric, not by the owner) -- every non-right-outlet
title on Trump's side, enforcement, immigration and the right's core causes,
whatever the outlet -- with border/immigration/ICE as the strictest area. The
labelled set is no longer restricted to the hostile outlets, and right-outlet
headlines on all of those topics are negatives (RIGHT_DOMAINS below picks
training negatives only; production never looks at the outlet), so the model
sees both sides of the immigration story. Cut 2026-10-06, measured after it,
word list included, one threshold for every source:
                                     @0.7   @0.8   @0.9   (old model @0.8: ~40%)
  labelled hostile                   54%    47%    35%
  labelled not hostile                6%     3%     1%
  unlabelled candidates             1.4%   0.7%   0.3%
  our own posts (288)                  7      4      3
  the ProPublica post                       0.91 (held out)
It still cannot tell direction on immigration: a right-outlet "Mamdani's Rage
at ICE ... Dishonest Grandstanding" scores 0.97, and only 14% of hostile ICE
headlines clear 0.8.

  ./.venv/bin/python tools/train_stance_vec.py [--cut 2026-10-03] --out models/stance_vec.json

Needs scikit-learn (training only; the guard reads the model with numpy)."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(".env")
from openai import OpenAI  # noqa: E402

SIDE = re.compile(r"\b(trump|maga|vance|hegseth|leavitt|ice|dhs|border patrol|patel|bondi|noem|miller|white house|"
                  r"republican|gop|rubio|bessent|lutnick|homan|administration)\b", re.I)
HOSTILE = re.compile(r"thedailybeast|huffpost|ms\.now|msnbc|theguardian|rollingstone|newrepublic|motherjones|salon\.com|"
                     r"thenation|independent\.co\.uk|vanityfair|rawstory|alternet|mediaite")
RIGHT = re.compile(r"gatewaypundit|breitbart|newsmax|dailycaller|townhall|redstate|twitchy|pjmedia|theblaze|westernjournal|"
                   r"dailywire|thefederalist|bizpacreview|100percentfedup|amgreatness|lifezette|wnd\.com|conservativetreehouse|"
                   r"revolver\.news|rsbnetwork|oann|louderwithcrowder|bearingarms|hotair|thepostmillennial|nypost|foxnews|"
                   r"washingtonexaminer|freebeacon|justthenews|trendingpolitics|conservativebrief|dailysignal|theepochtimes|instapundit")
WIRE = re.compile(r"apnews|reuters|thehill\.com|upi\.com|newsnationnow|axios")

LABELS = os.path.join("tools", "stance_labels.json")

# Hand-flagged hostile-frame posts (2026-10-06/07). Held out of training on purpose.
HOLDOUT = {
    "Trump Makes Natalie Harp, 35, Guest of Honor on Maiden Flight", "Trump Whisks Natalie Harp Away Again After Late Night",
    "Top MAGA Ally Delivers Jaw-Dropping Snub to Trump in Home State", "Must-Win GOP Senator Throws Trump Under the Bus in Tense Meeting",
    "Panicked Fox News Star Makes Blunt Demand as Blue Wave Looms", "Major White House Split Leaks as Aides Freak Out",
    "Must-Win MAGA Senate Candidate Caught Secretly Trashing Trump", "U.S. Troops Get on Hands and Knees to Welcome Communist Tyrant",
    "Republicans Rage as Trump Torpedoes Their Election Plan", "‘Emergency’ Bid to Stop Trump Slapping His Name on Another Landmark",
    "Keystone Kash’s FBI Humiliation Takes a Brutal New Turn", "Bombshell Drops in FBI’s Stephen Miller Phone Call Meltdown",
    "Trump Judge Shreds DHS Excuse for Arresting U.S. Citizen",
    "Press ban ruling shows Trump administration can’t just say ‘national security’ and win",
    "ICE hides locations of thousands of detainees with final removal orders",
    "Must-Win MAGA Candidate Roasted for Bonkers Debate Answer",
    "Electric-shock gloves ICE plans to use inflicted intense pain in army study",
    "As Trump’s Wall Closes in, DHS Intimidates a Border Tribe With Surveillance, Masked Agents and Drones",  # AI-50
}


# Right-wing outlets whose headlines are taken as negatives when unlabelled
# (2026-10-11): the in-use right-wing feeds plus right-wing sites seen in the
# published history. Training data selection only.
RIGHT_DOMAINS = (
    "100percentfedup.com",
    "aim.org",
    "alaskawatchman.com",
    "alphanews.org",
    "americanaffairsjournal.org",
    "americancompass.org",
    "americanmind.org",
    "amgreatness.com",
    "arizonadailyindependent.com",
    "army.mil",
    "bearingarms.com",
    "bizpacreview.com",
    "breitbart.com",
    "californiaglobe.com",
    "carolinajournal.com",
    "city-journal.org",
    "claremontreviewofbooks.com",
    "conservativebrief.com",
    "dailycaller.com",
    "dailycallernewsfoundation.org",
    "dailyhaymaker.com",
    "dailysignal.com",
    "dailytorch.com",
    "dailywire.com",
    "floridianpress.com",
    "flvoicenews.com",
    "foxnews.com",
    "freebeacon.com",
    "gatewayhispanic.com",
    "granitegrok.com",
    "hannity.com",
    "heartlandernews.com",
    "heritage.org",
    "hotair.com",
    "homeland.house.gov",
    "hughhewitt.com",
    "ijr.com",
    "instapundit.com",
    "judicialwatch.org",
    "justice.gov",
    "justthenews.com",
    "legalinsurrection.com",
    "libertynation.com",
    "lifesitenews.com",
    "lifezette.com",
    "louderwithcrowder.com",
    "nationalfile.com",
    "newsbusters.org",
    "nhjournal.com",
    "nypost.com",
    "oann.com",
    "patriotpost.us",
    "pjmedia.com",
    "powerlineblog.com",
    "presentdangerchina.org",
    "redstate.com",
    "revolver.news",
    "rsbnetwork.com",
    "spectator.org",
    "state.gov",
    "taxfoundation.org",
    "tennesseeconservativenews.com",
    "tennesseestar.com",
    "texasscorecard.com",
    "theamericanconservative.com",
    "theblaze.com",
    "thecentersquare.com",
    "thecollegefix.com",
    "theconservativetreehouse.com",
    "theepochtimes.com",
    "thefederalist.com",
    "thegatewaypundit.com",
    "thelibertydaily.com",
    "themainewire.com",
    "themidwesterner.news",
    "thenationalpulse.com",
    "thenevadaglobe.com",
    "thenewamerican.com",
    "theothermccain.com",
    "therightscoop.com",
    "thetexan.news",
    "thethinkingconservative.com",
    "townhall.com",
    "trendingpoliticsnews.com",
    "twitchy.com",
    "washingtonexaminer.com",
    "washingtonreporter.news",
    "washingtontimes.com",
    "westernjournal.com",
    "whitehouse.gov",
    "wisconsinrightnow.com",
    "wnd.com",
    "zerohedge.com",
    "rumble.com",
    "alexjoneslive.com",
    "americafirstreport.com",
    "slaynews.com",
    "rairfoundation.com",
    "stonezone.com",
    "thepostmillennial.com",
    "nataliegwinters.substack.com",
    "moonbattery.com",
    "americanthinker.com",
    "gatestoneinstitute.org",
    "notthebee.com",
    "modernity.news",
    "issuesinsights.com",
    "libertysentinel.org",
    "trevorloudon.com",
    "acecomments.mu.nu",
    "patriot.tv",
    "frontpagemag.com",
    "clashdaily.com",
    "rvmnews.com",
    "jdrucker.com",
    "liveaction.org",
    "independentsentinel.com",
    "newsmax.com",
    "nationalreview.com",
    "dhs.gov",
    "ice.gov",
    "cbp.gov",
    "defense.gov",
    "war.gov",
)

TOPIC = re.compile(
    r"\b(trump|maga|vance|hegseth|leavitt|ice|dhs|cbp|border|patel|fbi|bondi|doj|noem|miller|white house|"
    r"republicans?|gop|rubio|bessent|lutnick|homan|administration|deport\w*|immigra\w*|migrants?|illegals?|aliens?|"
    r"asylum|sanctuary|national guard|detention|detainees?|raids?|police|cops?|guns?|second amendment|abortion|"
    r"pro-life|trans\w*|gender|election|voter|ballot|christian\w*|israel\w*|jan\. 6|january 6|dei|woke|tariffs?|"
    r"musk|rfk|maha|conservatives?|right-wing|far-right|pentagon|troops|military)\b", re.I)


def _right(url: str) -> bool:
    from urllib.parse import urlparse
    host = (urlparse(url or "").hostname or "").lower().removeprefix("www.")
    return bool(RIGHT.search(url)) or any(host == d or host.endswith("." + d) for d in RIGHT_DOMAINS)


def collect(cut_ts: float) -> list[tuple[str, int]]:
    right = _right
    hand = json.load(open(LABELS, encoding="utf-8"))
    rows: dict[str, int] = {}
    for line in open("logs/prescore_decisions.jsonl", encoding="utf-8"):
        r = json.loads(line)
        t, u = r["title"].strip(), r["url"]
        if r["logged_at"] >= cut_ts or t in HOLDOUT or t in rows or len(t) < 20:
            continue
        if t in hand:
            rows[t] = hand[t]
        elif (right(u) or WIRE.search(u)) and TOPIC.search(t):
            rows[t] = 0
    for line in open("logs/engagement_snapshots.jsonl", encoding="utf-8"):
        r = json.loads(line)
        t = (r.get("ttl") or "").strip()
        if not t or t in HOLDOUT or r["cdate"] / 1000 >= cut_ts:
            continue
        if t in hand:
            rows[t] = hand[t]
        elif t not in rows and right(r.get("src") or ""):
            rows[t] = 0
    return list(rows.items())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", default="2026-10-03")
    ap.add_argument("--out", default="models/stance_vec.json")
    ap.add_argument("--threshold", type=float, default=0.8)
    a = ap.parse_args()
    cut = dt.datetime.fromisoformat(a.cut).replace(tzinfo=dt.timezone.utc).timestamp()
    data = collect(cut)
    titles = [t for t, _ in data]
    client = OpenAI()
    vecs = []
    for i in range(0, len(titles), 500):
        vecs += [e.embedding for e in client.embeddings.create(model="text-embedding-3-small", input=titles[i:i + 500]).data]
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced")
    clf.fit(np.array(vecs), [y for _, y in data])
    model = {
        "version": "stance-vec-" + a.cut.replace("-", "") + "-hand2",
        "trained_at": int(dt.datetime.now(dt.timezone.utc).timestamp()),
        "embedding_model": "text-embedding-3-small", "input": "title",
        "n_train": len(data), "n_hostile": sum(y for _, y in data), "cut": a.cut,
        "threshold": a.threshold,
        "w": clf.coef_[0].tolist(), "b": float(clf.intercept_[0]),
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(model, fh)
    print("wrote %s: %d titles (%d hostile), threshold %.2f" % (a.out, len(data), model["n_hostile"], a.threshold))


if __name__ == "__main__":
    main()
