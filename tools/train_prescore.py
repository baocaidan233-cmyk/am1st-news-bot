"""Train the Layer 1.55 pre-score model — see core/prescore.py.

TWO LABEL SOURCES, and --labels-from is REQUIRED so that the choice is always
made on purpose. There is no default, because the two are not interchangeable
and a silent fallback is how a retrain would quietly undo a fix:

  --labels-from log      the historical source, described below. Negatives are
                         the Scorer's own rejection lines in main.log.
  --labels-from offline  a file of {title: {g,d,who,drop}} answers produced by
                         running the live scoring prompt over the candidate
                         stream offline, joined to logs/prescore_decisions.jsonl
                         for the url and timestamp. Needs --labels PATH.

Why "offline" exists (2026-10-02). The "log" source has a hole that cannot be
closed from inside it: a candidate this very model SKIPS never reaches the
Scorer, so it has no score and no rejection line, so it is absent from training.
The model therefore cannot see the cases where it is wrong. The audit cohort
(prescore.audit_rate) covers that region at 10% with inverse-propensity
weighting, which is why it exists — but 10% of the region where the errors live
is thin. The "offline" source labels EVERY row including the skipped ones, so no
reweighting is needed and the error region is fully represented; inverse-
propensity weighting is therefore switched off in that mode, and applying both
would double-count.

The hole was not theoretical. The model shipped 2026-09-30 was trained against
the retired 21-theme scoring prompt, and when production moved to
prompts/scoring_prompt_v2.txt on 2026-10-02 it kept skipping what the retired
prompt would have refused — including the Lindsay Clancy retrial, which v2
scores 6.0 and would publish, at p=0.0467 against a 0.0649 threshold. Retrained
on offline labels, the same title comes out at p=0.7711.

Labels for --labels-from log come from two places because this channel keeps
them in two places:
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

# Two shapes, because main.py's rejection line gained a field on 2026-10-02:
#   before:  run_cycle: <url> scored 4.0, below threshold — <title>
#   after:   run_cycle: <url> scored 4.0, below threshold — <llm_comment> — <title>
# Both must parse, or a retrain silently trains on a mixture of titles and
# "comment — title" strings and the embeddings stop meaning anything.
_REJECTED = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ INFO main: run_cycle: (\S+) scored ([\d.]+), "
    r"below threshold — (.+)$")


def _title_from_tail(tail: str) -> str:
    """The comment, when present, is the Scorer's llm_comment and always carries
    a "field=value" pair; a real headline does not. Splitting on that rather
    than on the separator keeps both log shapes readable."""
    if " — " in tail:
        head, rest = tail.split(" — ", 1)
        if "=" in head:
            return rest.strip()
    return tail.strip()


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
            stamp, url, score, tail = m.groups()
            title = _title_from_tail(tail)
            if url in seen or not title:
                continue
            seen.add(url)
            rows.append({"ts": int(datetime.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
                                   .replace(tzinfo=datetime.timezone.utc).timestamp()),
                         "url": url, "title": title, "llm_score": float(score)})
    return rows


def offline_rows(labels_path: str) -> list[dict]:
    """Rows from an offline labelling pass. Every candidate that reached Layer
    1.55 is here, skipped ones included, which is the whole point of this source.

    The labels file maps title -> the four closed-set fields exactly as
    core.scoring.normalise returns them; the score is recomputed here with
    core.scoring.compute_score rather than stored, so this can never drift from
    what the live Scorer would have produced from the same fields."""
    from core.scoring import compute_score

    with open(labels_path, encoding="utf-8") as fh:
        labels = json.load(fh)
    rows, seen = [], set()
    with open("logs/prescore_decisions.jsonl", encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            title = (d.get("title") or "").strip()
            url = d.get("url")
            if not title or not url or url in seen:
                continue
            fields = labels.get(title)
            if not fields:
                continue
            seen.add(url)
            rows.append({"ts": int(d["logged_at"]), "url": url, "title": title,
                         "llm_score": compute_score(fields, title)})
    rows.sort(key=lambda r: r["ts"])
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


def audit_decisions(config) -> dict[str, str]:
    """url -> the pre-score verdict, for rows the audit cohort forced through.

    Once the layer is live, a candidate it skipped is never scored and so can
    never appear in the labels above. What survives is the set the model was
    already willing to pass, and training on that alone lets the model confirm
    its own judgement: the kinds of story it skips stop appearing, so the next
    model skips them harder. The audit cohort is the only slice not filtered
    by the model, and the ones inside it the model wanted to skip are the only
    rows that can argue back. They stand in for every skipped candidate, so
    they are weighted by the reciprocal of the cohort's share."""
    out: dict[str, str] = {}
    for path in sorted(glob.glob(config.prescore.log_path + "*")):
        for line in _lines(path):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("url") and r.get("decision"):
                out[r["url"]] = r["decision"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/prescore.json")
    ap.add_argument("--version", default=time.strftime("prescore-%Y%m%d"))
    ap.add_argument("--keep-min", type=float, default=None,
                    help="threshold keeps every training row scoring at least this "
                         "(default: config openai.score_threshold)")
    # Required on purpose — see the module docstring. A default here is how a
    # later retrain would silently go back to the source with the blind spot.
    ap.add_argument("--labels-from", choices=("log", "offline"), required=True,
                    help="log: Scorer rejection lines + Notion. "
                         "offline: --labels file covering every candidate, skipped included.")
    ap.add_argument("--labels", default=None,
                    help="with --labels-from offline: {title: {g,d,who,drop}} JSON")
    args = ap.parse_args()
    if args.labels_from == "offline" and not args.labels:
        ap.error("--labels-from offline needs --labels PATH")

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

    if args.labels_from == "offline":
        rows = offline_rows(args.labels)
        if not rows:
            sys.exit(f"no rows matched between {args.labels} and logs/prescore_decisions.jsonl")
        first_negative = min(r["ts"] for r in rows)
    else:
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

    # Inverse-propensity weighting corrects for the "log" source seeing the
    # skip region only through the audit cohort. The "offline" source labels
    # that region in full, so weighting it again would double-count it.
    if args.labels_from == "offline":
        weights = None
        n_reweighted = 0
        print("inverse-propensity weighting: off — every candidate is labelled, "
              "including the ones the model would have skipped")
    else:
        verdicts = audit_decisions(config)
        rate = max(config.prescore.audit_rate, 1e-6)
        weights = np.array([1.0 / rate if verdicts.get(r["url"]) == "audit_would_skip" else 1.0
                            for r in rows])
        n_reweighted = int((weights > 1).sum())
        print(f"inverse-propensity weighting: {n_reweighted} rows the model wanted to skip and the "
              f"audit cohort kept, weighted {1/rate:.0f}x")

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

    def fit(Xs, ys, protect, ws=None):
        model = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
        oof = cross_val_predict(model, Xs, ys, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                method="predict_proba",
                                params={"sample_weight": ws} if ws is not None else None)[:, 1]
        model.fit(Xs, ys, sample_weight=ws)
        return model, (float(np.min(oof[protect])) if protect.any() else 0.0), oof

    _, _, oof_all = fit(X, y, keep & judgeable, weights)
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
        # weights is None in offline mode, where every row is labelled and
        # nothing needs reweighting — so it cannot be sliced like an array.
        model, thr, _ = fit(X[tr], y[tr], (keep & judgeable)[tr],
                            None if weights is None else weights[tr])
        skip = (model.predict_proba(X[te])[:, 1] < thr) & judgeable[te]
        lost = []
        for gate in (5.0, 6.0, 7.0):
            yg = np.array([r["llm_score"] >= gate for r in rows])[te]
            lost.append(f"{int((skip & yg).sum())}/{int(yg.sum())}")
        print(f"{tr.sum():>7} {te.sum():>7} {skip.mean():>8.1%} {lost[0]:>10} {lost[1]:>10} {lost[2]:>10}")

    model, threshold, oof = fit(X, y, keep & judgeable, weights)
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
        "n_reweighted": n_reweighted,
        "audit_rate": config.prescore.audit_rate,
        "n_at_gate": int(keep.sum()),
        "gate": keep_min,
        "target": "score>=5",
        "labels_from": args.labels_from,
        "labels_file": args.labels if args.labels_from == "offline" else None,
        "scoring_prompt": config.openai.scoring_prompt_file,
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
