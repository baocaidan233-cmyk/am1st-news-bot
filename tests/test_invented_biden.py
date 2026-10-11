"""core/caption_guard.invented_biden: a caption may name Biden only if its source does (2026-10-11)."""
from core.caption_guard import invented_biden

SRC = "x" * 500


def test_named_without_source():
    # AM1ST 10/11, ProPublica on the Tohono O'odham Nation: the source never names him
    assert invented_biden("Crossings fell 94% compared to the Biden era.", "ProPublica. " + SRC) == "invented-biden"


def test_named_with_source_in_any_script():
    assert invented_biden("Under the Biden administration ...", "during the Biden administration " + SRC) is None
    assert invented_biden("Biden said ...", "Байден заявил " + SRC) is None
    assert invented_biden("Biden said ...", "拜登表示 " + SRC) is None


def test_too_little_source_to_judge():
    assert invented_biden("President Biden said", "short") is None
