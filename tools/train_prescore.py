"""Train the Layer 1.55 pre-score model — see core/prescore.py.

Labels come from two places because this channel keeps them in two places:
a candidate the Scorer rejected leaves a line in main.log carrying its URL,
its score and (since 2026-09-28) its title, and one it accepted is a row in
the Notion candidate database. Neither half alone can train this: Notion has
only positives, and before 2026-09-28 the log line had no title.

That split is also why this trains on a window rather than on everything.
The negatives start when the title was added to the log line, so rows older
than the first negative are dropped -- keeping them would hand the model a
stretch of history containing positives and no negatives, and it would learn
the calendar.

The threshold is the lowest out-of-fold probability among training rows at
or above the gate, so the model would not have skipped a single one of them.
It is a minimum, not a percentile: one mislabelled row can only make it
lower and skip less, never make it skip something it should have kept.

Run:  ./.venv/bin/python tools/train_prescore.py --out models/prescore.json
      (needs scikit-learn, which only training needs — core/prescore.py
      reads the model with numpy alone.)
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import glob
import gzip
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# "run_cycle: <url> scored 4.0, below threshold — <title>"
_REJECTED = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ INFO main: run_cycle: (\S+) scored ([\d.]+), "
    r"below threshold — (.+)$")


def _lines(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as fh:
        for line in fh:
            yield line.rstrip("\n")


def rejected_rows() -> list[dict]:
    rows, seen = [], set()
    for path in sorted(glob.glob("logs/main.log*")):
        for line in _lines(path):
            m = _REJECTED.match(line)
            if not m:
                continue
            stamp, url, score, title = m.groups()
            if url in seen or not title.strip():
                continue
            seen.add(url)
            rows.append({"ts": int(datetime.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
                                   .replace(tzinfo=datetime.timezone.utc).timestamp()),
                         "url": url, "title": title.strip(), "llm_score": float(score)})
    return rows


async def accepted_rows(config) -> list[dict]:
    import httpx
    props = config.notion.candidate_props
    key = os.environ.get("NOTION_CANDIDATE_API_KEY") or os.environ["NOTION_API_KEY"]
    out, cursor = [], None
    async with httpx.AsyncClient(timeout=60, headers={
            "Authorization": f"Bearer {key}", "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"}) as client:
        while True:
            body = {"page_size": 100,
                    "filter": {"property": props.llm_score, "number": {"greater_than_or_equal_to": 0}}}
            if cursor:
                body["start_cursor"] = cursor
            resp = await client.post(
                f"https://api.notion.com/v1/databases/{config.notion.candidate_db_id}/query", json=body)
            resp.raise_for_status()
            page = resp.json()
            for row in page.get("results", []):
                p = row.get("properties", {})
                title_parts = p.get(props.title, {}).get("title") or []
                url_prop = p.get(props.url, {})
                url = url_prop.get("url") or "".join(
                    x.get("plain_text", "") for x in (url_prop.get("rich_text") or []))
                score = p.get(props.llm_score, {}).get("number")
                if not title_parts or score is None or not url:
                    continue
                out.append({"ts": int(datetime.datetime.fromisoformat(
                                row["created_time"].replace("Z", "+00:00")).timestamp()),
                            "url": url,
                            "title": "".join(x.get("plain_text", "") for x in title_parts),
                            "llm_score": float(score)})
            cursor = page.get("next_cursor")
            if not page.get("has_more"):
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/prescore.json")
    ap.add_argument("--version", default=time.strftime("prescore-%Y%m%d"))
    ap.add_argument("--keep-min", type=float, default=None,
                    help="threshold keeps every training row scoring at least this "
                         "(default: config openai.score_threshold)")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from openai import OpenAI
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    from core.config import load_config
    from core.prescore import title_word_count

    config = load_config("config/config.yaml")
    keep_min = args.keep_min if args.keep_min is not None else config.openai.score_threshold

    rejected = rejected_rows()
    if not rejected:
        sys.exit("no rejected rows in logs/main.log* — is the title still being logged with the score?")
    accepted = asyncio.run(accepted_rows(config))
    first_negative = min(r["ts"] for r in rejected)
    seen = {r["url"] for r in rejected}
    rows = rejected + [r for r in accepted if r["ts"] >= first_negative and r["url"] not in seen]
    rows.sort(key=lambda r: r["ts"])
    if len(rows) < 500:
        sys.exit(f"only {len(rows)} labelled rows — not enough to train")
    print(f"{len(rows)} rows from {datetime.datetime.utcfromtimestamp(first_negative):%Y-%m-%d %H:%M} UTC "
          f"({sum(1 for r in rows if r['llm_score'] >= keep_min)} at or above the {keep_min:.0f} gate)")
    print("scores:", dict(sorted(Counter(r["llm_score"] for r in rows).items())))

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    vectors: list = []
    for i in range(0, len(rows), 256):
        chunk = [(r["title"] or " ")[:500] for r in rows[i:i + 256]]
        resp = client.embeddings.create(model="text-embedding-3-small", input=chunk)
        vectors += [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
    X = np.asarray(vectors, dtype=np.float32)

    # The model predicts "reaches 5", which is where the pool gate sits and
    # where the positives are; the THRESHOLD is what protects keep_min.
    y = np.array([r["llm_score"] >= 5 for r in rows])
    keep = np.array([r["llm_score"] >= keep_min for r in rows])
    # A title this short goes to the Scorer whatever the model says, so it
    # must not set the threshold either — that is the whole point of it.
    judgeable = np.array([title_word_count(r["title"]) > config.prescore.min_title_words for r in rows])
    ts = np.array([r["ts"] for r in rows])

    def fit(Xs, ys, protect):
        model = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
        oof = cross_val_predict(model, Xs, ys, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                method="predict_proba")[:, 1]
        model.fit(Xs, ys)
        return model, (float(np.min(oof[protect])) if protect.any() else 0.0), oof

    _, _, oof_all = fit(X, y, keep & judgeable)
    print(f"out-of-fold AUC: >=5 {roc_auc_score(y, oof_all):.4f}", end="")
    for gate in (6.0, 7.0):
        yg = np.array([r["llm_score"] >= gate for r in rows])
        if yg.sum() > 5:
            print(f"   >={gate:.0f} {roc_auc_score(yg, oof_all):.4f}", end="")
    print()

    # Time-ordered holdouts: train on the earliest share, measure on the rest.
    # Four splits of one night are not four independent experiments, so read
    # the zero-loss rows as "nothing seen yet", not as a proven miss rate.
    print(f"\n{'train':>7} {'test':>7} {'skipped':>9} {'>=5 lost':>10} {'>=6 lost':>10} {'>=7 lost':>10}")
    for q in (0.4, 0.5, 0.6, 0.7):
        cut = np.quantile(ts, q)
        tr, te = ts < cut, ts >= cut
        if tr.sum() < 200 or te.sum() < 100:
            continue
        model, thr, _ = fit(X[tr], y[tr], (keep & judgeable)[tr])
        skip = (model.predict_proba(X[te])[:, 1] < thr) & judgeable[te]
        lost = []
        for gate in (5.0, 6.0, 7.0):
            yg = np.array([r["llm_score"] >= gate for r in rows])[te]
            lost.append(f"{int((skip & yg).sum())}/{int(yg.sum())}")
        print(f"{tr.sum():>7} {te.sum():>7} {skip.mean():>8.1%} {lost[0]:>10} {lost[1]:>10} {lost[2]:>10}")

    model, threshold, oof = fit(X, y, keep & judgeable)
    skipped = ((model.predict_proba(X)[:, 1] < threshold) & judgeable).mean()
    print(f"\nfull fit: threshold {threshold:.4f}, would skip {skipped:.1%} of these rows")

    # Out-of-distribution reference: every training title's unit vector, int8,
    # and the 5th percentile of their nearest-neighbour similarities. A title
    # less like anything here than that is never skipped — a story with no
    # precedent in the training window is exactly what a model fitted on that
    # window has no standing to rule out.
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    nearest = np.empty(len(Xn), dtype=np.float32)
    for i in range(0, len(Xn), 1000):
        sims = Xn[i:i + 1000] @ Xn.T
        for j in range(sims.shape[0]):
            sims[j, i + j] = -1.0
        nearest[i:i + 1000] = sims.max(axis=1)
    ood_cut = float(np.percentile(nearest, 5))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ref_name = out.stem + "_ref.npy"
    np.save(out.parent / ref_name, np.clip(np.rint(Xn * 127), -127, 127).astype(np.int8))
    out.write_text(json.dumps({
        "version": args.version,
        "trained_at": int(time.time()),
        "n_train": len(rows),
        "n_at_gate": int(keep.sum()),
        "gate": keep_min,
        "target": "score>=5",
        "threshold_rule": f"min out-of-fold p among score>={keep_min:.0f} with a judgeable title",
        "embedding_model": "text-embedding-3-small",
        "input": "title[:500]",
        "min_title_words": config.prescore.min_title_words,
        "threshold": threshold,
        "b": float(model.intercept_[0]),
        "w": [float(x) for x in model.coef_[0]],
        "ood_ref_file": ref_name,
        "ood_cut": ood_cut,
    }), encoding="utf-8")
    print(f"wrote {out} and {out.parent / ref_name}")


if __name__ == "__main__":
    main()
