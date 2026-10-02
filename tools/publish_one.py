#!/usr/bin/env python3
"""Publish one candidate now, by URL, instead of waiting for it to win a slot.

    .venv/bin/python tools/publish_one.py <url> [--dry]

Why this exists. A candidate can be right, fresh, and high-scoring and still
never go out: measured 2026-10-02 over 14 days, the publish rate is 48% in the
7.0 and 8.0 tiers, and the mechanism is visible in the ranking log — a candidate
that loses a batch keeps ageing, the freshness penalty grows with age (median
1.63 at the 8.0 tier against 1.10 at 6.0), and losing makes it likelier to lose
again. 76 candidates at 7.0+ matured past the publish window unpublished, and
only 2.6% of them were stories we had already covered.

The gates are the ones main_publish.py runs, in its order, and none of them is
skipped — that is the whole point of this file existing rather than a hand-made
post:

    extractor.extract        no publishing off title+description alone
    is_english               on the extracted body, not the feed
    writer.write             the real caption prompt, not hand-written text
    is_no_comment            the writer's own refusal is honoured
    caption_guard            "former president" can never go out
    title_guard              every title in the caption must match the source
    fetch_link_preview       plus core/title_tail.py on the card headline
    card_for_broken_preview  our own card when Gettr's preview is unusable
    publisher.publish
    mark_send_status         REQUIRED. Without it the normal cycle publishes
                             the same candidate again — the row stays eligible.

Run it with --dry first. It stops before publishing and prints the exact caption,
which is the last chance to read what is about to go out under the channel's name.
"""
from __future__ import annotations
import asyncio, os, sys
sys.path.insert(0, os.path.expanduser("~/AM1ST")); os.chdir(os.path.expanduser("~/AM1ST"))
from dotenv import load_dotenv; load_dotenv(".env")
from core.config import load_config
from core.notion_candidates import query_eligible_candidates, mark_send_status
from agents.extractor import Extractor
from agents.writer import Writer
from agents.gettr_publisher import GettrPublisher
from agents.og_metadata import fetch_link_preview
from core.caption_guard import former_president_violation
from core.title_guard import title_violation
from core.language import is_english
import main_publish as MP

TARGET = sys.argv[1]
DRY = "--dry" in sys.argv

async def main():
    cfg = load_config("config/config.yaml")
    norm = lambda u: (u or "").split("?")[0].rstrip("/")
    pool = await query_eligible_candidates(cfg)
    hit = [c for c in pool if norm(c.url) == norm(TARGET)]
    if not hit:
        print(f"候选池里没有 {TARGET}\n池子里共 {len(pool)} 条"); return
    c = hit[0]
    print(f"找到: {c.llm_score:.1f}分  {c.title[:80]}")
    alerts = None
    try:
        from core.alerts import Alerts
        alerts = Alerts(cfg)
    except Exception:
        pass
    extractor = Extractor(cfg, alerts)
    writer = Writer(cfg)

    sources = {}
    try:
        sources = MP.load_sources(cfg) if hasattr(MP, "load_sources") else {}
    except Exception:
        sources = {}
    text = await extractor.extract(c.url, sources, attempts=getattr(c, "extraction_attempts", 0))
    if not text:
        print("✗ 正文抓取失败 —— 按规矩不拿标题+摘要硬发"); return
    c.content = text
    print(f"✓ 正文 {len(text)} 字符")
    if not is_english(c.content[:1000]):
        print("✗ 正文非英语，丢弃"); return
    post_content = await writer.write(c.title, c.content, context="",
                                      is_opinion=False, published_at=c.published_at)
    if Writer.is_no_comment(post_content):
        print("✗ 写手返回 No comment"); return
    v = former_president_violation(post_content)
    if v:
        print(f"✗ caption_guard 拦下: {v}"); return
    t = title_violation(post_content, c.content or "")
    if t:
        print(f"✗ title_guard 拦下: {t}"); return
    print("✓ 两道闸门通过")
    c.post_content = f"{post_content}\n\n{c.url}"
    print("\n--- 要发的文案 ---")
    print(c.post_content)
    print("--- 完 ---\n")
    og = await fetch_link_preview(c.url)
    print(f"预览: ttl={ (og.get('prev_ttl') or '')[:70]!r}")
    print(f"      img={'有' if og.get('prev_img') else '无'}")
    card = await MP.card_for_broken_preview(cfg, title=og.get("prev_ttl") or c.title or "",
                                            source_url=c.url, image_url=og.get("prev_img"),
                                            dry_run=DRY)
    if DRY:
        print("\n[dry] 到此为止，不发布"); return
    publisher = GettrPublisher(cfg, dry_run=False)
    post_id = await publisher.publish(
        c.post_content, log_ref=c.url,
        prev_desc=og.get("prev_desc") or c.description or None,
        prev_img=og.get("prev_img"),
        prev_src_link=og.get("prev_src_link") or c.url,
        prev_ttl=og.get("prev_ttl") or c.title,
        media=card)
    if post_id:
        print(f"\n✓ 已发布 https://gettr.com/post/{post_id}")
        ok = await mark_send_status(cfg, c.page_id)
        print(f"  标记已发: {'成功' if ok else '失败 —— 要手工确认，否则会重发'}")
    else:
        print("\n✗ 发布失败")
asyncio.run(main())
