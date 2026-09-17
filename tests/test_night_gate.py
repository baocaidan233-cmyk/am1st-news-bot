"""深夜门控纯逻辑测试 —— 不触网、不跑 bot 本体。"""
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from core.config import AppConfig
from core.models import PublishCandidate
from agents.candidate_selector import is_night, select_batch

ET = ZoneInfo("America/New_York")
cfg = AppConfig()
fails = []

def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got={got} want={want}")
    if not ok: fails.append(name)

def et(y, m, d, h):
    return datetime(y, m, d, h, 30, tzinfo=ET).astimezone(timezone.utc)

print("== is_night 边界 (默认 0-7 ET) ==")
for h, want in [(0,True),(3,True),(6,True),(7,False),(12,False),(19,False),(23,False)]:
    check(f"ET {h:02d}:30", is_night(et(2026,9,17,h), cfg), want)

print("== 夏令时/冬令时都按美东当地小时判断 ==")
check("7月(EDT) 02:30 ET", is_night(et(2026,7,15,2), cfg), True)
check("1月(EST) 02:30 ET", is_night(et(2026,1,15,2), cfg), True)
check("7月(EDT) 09:30 ET", is_night(et(2026,7,15,9), cfg), False)

print("== 跨午夜窗口 + 关闭开关 ==")
c2 = AppConfig(); c2.publish.night_start_hour, c2.publish.night_end_hour = 22, 6
for h, want in [(22,True),(23,True),(0,True),(5,True),(6,False),(12,False),(21,False)]:
    check(f"22-6 窗口 ET {h:02d}", is_night(et(2026,9,17,h), c2), want)
c3 = AppConfig(); c3.publish.night_start_hour = c3.publish.night_end_hour = 0
check("start==end 关闭窗口 (ET 03)", is_night(et(2026,9,17,3), c3), False)

print("== select_batch 深夜硬过滤 ==")
def cand(pid, score, hours_old, now, hot=False):
    return PublishCandidate(
        page_id=pid, title=f"t{pid}", url=f"https://x.test/{pid}",
        post_content="body", llm_score=score,
        published_at=now - timedelta(hours=hours_old), created_at=now, is_hot=hot,
    )

now_night = et(2026, 9, 17, 3)     # 周四 03:30 ET
now_day   = et(2026, 9, 17, 14)    # 周四 14:30 ET

pool = [cand("a",6.0,1,now_night), cand("b",6.0,2,now_night),
        cand("c",7.0,1,now_night), cand("d",8.0,2,now_night),
        cand("e",9.0,3,now_night)]

import agents.candidate_selector as cs
real_now = cs.datetime
class FakeDT:
    @staticmethod
    def now(tz=None): return now_night
    def __getattr__(self, k): return getattr(real_now, k)
cs.datetime = FakeDT()
b = select_batch(list(pool), cfg)
cs.datetime = real_now
floor = cfg.publish.night_min_score
want = sorted(c.page_id for c in pool if c.llm_score >= floor)
check(f"深夜只留 >={floor} 的稿", sorted(c.page_id for c in b), want)

cs.datetime = FakeDT()
low = cfg.publish.night_min_score - 1.0
b2 = select_batch([cand("a",low,1,now_night), cand("b",low-1,2,now_night)], cfg)
cs.datetime = real_now
check(f"深夜全部低于 {cfg.publish.night_min_score} -> 空批 (tier5 兜底也不能绕过)", [c.page_id for c in b2], [])

cs.datetime = FakeDT()
b3 = select_batch([cand("a",low,1,now_night), cand("h",5.0,2,now_night,hot=True)], cfg)
cs.datetime = real_now
check("深夜 is_hot 豁免 (人工标记的突发仍可发)", [c.page_id for c in b3], ["h"])

class FakeDTDay:
    @staticmethod
    def now(tz=None): return now_day
    def __getattr__(self, k): return getattr(real_now, k)
# 白天组必须用白天的基准时间造稿，否则会撞上 12h 上限（那是既有规则，不是本次改动）
pool_day = [cand("a",6.0,1,now_day), cand("b",6.0,2,now_day),
            cand("c",7.0,1,now_day), cand("d",8.0,2,now_day),
            cand("e",9.0,3,now_day)]
cs.datetime = FakeDTDay()
b4 = select_batch(list(pool_day), cfg)
cs.datetime = real_now
check("白天不受影响 (6.0 仍可入选)", sorted(c.page_id for c in b4), ["a","b","c","d","e"])

cs.datetime = FakeDTDay()
b5 = select_batch([cand("lo",5.0,1,now_day)], cfg)
cs.datetime = real_now
check("白天 5.0 仍走既有 weekday->weekend 放宽 (未被深夜逻辑影响)", [c.page_id for c in b5], ["lo"])

print("\n" + ("全部通过" if not fails else f"失败 {len(fails)} 项: {fails}"))
sys.exit(1 if fails else 0)
