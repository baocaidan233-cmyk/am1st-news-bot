from __future__ import annotations

import logging

from core.config import AppConfig

logger = logging.getLogger(__name__)


def compute_adjustments(config: AppConfig, recent_counts: dict[str, int]) -> dict[str, float]:
    """Turns "what we have actually published lately" into a per-topic score
    adjustment, as a saturating proportional controller:

        adjustment = clamp(gain * (target_share - actual_share), ±max_adjustment)

    This is deliberately NOT a fixed per-topic bonus. A fixed bonus has no
    self-limiting term: the favoured topic wins every tie it is in, its share
    climbs without bound, and the channel converges on one subject. Two
    separate measurements say that is the wrong direction — 2026-09-25 found
    that topic repetition inside a rolling window LOWERS breakout rate
    (0.71x, p=0.0082, pooled over 13 channels; GlobalPulse 0.08x), and the
    supply is thin enough that quadrupling any one topic means taking its
    worst stories instead of its best. Here, a topic that runs ahead of its
    target has its own adjustment go negative and yields the next slot back.
    It cannot monopolise the feed no matter how attractive its target is.

    `targets` is ordinary config, not a learned parameter. It is where an
    editorial judgment about what this channel should cover belongs — if a
    subject matters for reasons engagement cannot see, raise its target and
    the controller simply carries it out.

    Returns {} when disabled, or until the window holds min_window_posts —
    every caller treats {} as "no adjustment anywhere", i.e. exactly today's
    behaviour. That floor matters more than it looks: with an empty window
    every topic reads as maximally under-target, and the largest target
    ("其他", a catch-all) would saturate at +max and take the whole batch.
    A short window is nearly as bad, since actual shares can then only take
    a few coarse values and the controller swings between extremes. So below
    the floor it does nothing at all, which is also the correct behaviour on
    a cold start and after any outage.

    An unknown/untagged topic never appears in the result, so it gets 0.0
    from callers' .get(topic, 0.0)."""
    mix = config.topic_mix
    if not mix.enabled or not mix.targets:
        return {}

    total = sum(recent_counts.values())
    if total < mix.min_window_posts:
        logger.info(
            "topic_mix: window holds %d published posts, below min_window_posts=%d — no adjustment this cycle",
            total, mix.min_window_posts,
        )
        return {}

    adjustments: dict[str, float] = {}
    for topic, target in mix.targets.items():
        actual = (recent_counts.get(topic, 0) / total) if total else 0.0
        raw = mix.gain * (target - actual)
        adjustments[topic] = max(-mix.max_adjustment, min(mix.max_adjustment, raw))

    over = sorted(((t, a) for t, a in adjustments.items() if a < 0), key=lambda z: z[1])[:3]
    under = sorted(((t, a) for t, a in adjustments.items() if a > 0), key=lambda z: -z[1])[:3]
    logger.info(
        "topic_mix: window has %d published posts; most over-target %s; most under-target %s",
        total,
        [(t, round(a, 2)) for t, a in over],
        [(t, round(a, 2)) for t, a in under],
    )
    return adjustments
