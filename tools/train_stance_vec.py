"""Train the vector stance check -- see core/stance_guard.StanceVectorGuard.

What it learns is whose side a headline is written from, not whether it is
negative. Labels come from the outlet, by distant supervision, and the outlet is
never an input to the model:

  1  a left-hostile outlet's headline about Trump's side (Daily Beast, HuffPost,
     MS NOW, the Guardian, Rolling Stone, The New Republic ...)
  0  a right-leaning outlet's headline about the same side, a wire service's,
     and every title this channel has published that did not come from one of
     the hostile outlets above

A titles-only logistic regression on text-embedding-3-small, the same embedding
model the pipeline already uses.

The first version (2026-10-07) was trained on everything logged before
2026-10-03 00:00 UTC and measured on what came after. At the deployed 0.8
threshold: 49% of hostile-outlet headlines in the test period flagged (the word
list: 12.9%); 3.6% of right-outlet and 5.6% of wire headlines; and of the 210
posts the channel actually published after the cut, 4 -- all four of them
hostile-frame posts. Hand-flagged errors are never training data: titles listed
in HOLDOUT stay out so they remain an honest check.

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
}


def collect(cut_ts: float) -> list[tuple[str, int]]:
    rows: dict[str, int] = {}
    for line in open("logs/prescore_decisions.jsonl", encoding="utf-8"):
        r = json.loads(line)
        t, u = r["title"].strip(), r["url"]
        if r["logged_at"] >= cut_ts or t in HOLDOUT or t in rows or len(t) < 20 or not SIDE.search(t):
            continue
        if HOSTILE.search(u):
            rows[t] = 1
        elif RIGHT.search(u) or WIRE.search(u):
            rows[t] = 0
    for line in open("logs/engagement_snapshots.jsonl", encoding="utf-8"):
        r = json.loads(line)
        t = (r.get("ttl") or "").strip()
        if t and t not in rows and t not in HOLDOUT and r["cdate"] / 1000 < cut_ts and not HOSTILE.search(r.get("src") or ""):
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
    clf = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced")
    clf.fit(np.array(vecs), [y for _, y in data])
    model = {
        "version": "stance-vec-" + a.cut.replace("-", ""),
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
