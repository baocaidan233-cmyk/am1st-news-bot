"""core/time_words.py (from DailyNews) on cases from this channel group's own posts, 2026-10-11."""
import datetime as dt

from core.time_words import fix_caption_time_words, fix_time_words

UTC = dt.timezone.utc
LONG = "x" * 400        # a body long enough to count as the article


def test_the_source_form_and_eastern():
    # Ibovespa (MW): the source says quinta-feira; the post's "October 8" becomes Thursday
    post, ch = fix_time_words("The Ibovespa rose 1.02% on October 8, driven by oil.",
                              "Ibovespa sobe\nO Ibovespa encerrou a quinta-feira em alta.",
                              dt.datetime(2026, 10, 8, 21, 9, tzinfo=UTC), "https://www.infomoney.com.br/x")
    assert "on Thursday" in post, (post, ch)


def test_dates_in_other_languages_and_numbers_are_not_cut():
    for src, day in (("Поздним вечером 9 октября президент России поговорил", "October 9"),
                     ("Un atac cu drone a provocat vineri, 9 octombrie, perturbări", "October 9"),
                     ("по состоянию на 2 октября увеличилась | 09.10.2026, ПРАЙМ", "October 2"),
                     ("@Rasmussen_Poll | 9/24-30 | 1,794 LV", "September 24")):
        post = f"It happened on {day}, officials said."
        out, ch = fix_time_words(post, "T\n" + src, dt.datetime(2026, 10, 10, 1, 0, tzinfo=UTC), "https://www.example.com/x")
        assert day in out, (src, out, ch)


def test_a_rocket_is_not_a_date():
    post = "China launched 27 satellites on a Long March 12 rocket on October 10."
    out, ch = fix_time_words(post, "T\n北京时间10月10日3时27分，长征十二号运载火箭",
                             dt.datetime(2026, 10, 9, 23, 48, tzinfo=UTC), "https://sputniknews.cn/x")
    assert out == post, (out, ch)


def test_a_bare_day_range_stands():
    post = "Trade talks are set for Beijing on October 8-9."
    out, ch = fix_time_words(post, "T\n（中央社台北7日電）歐盟貿易專員8日至9日前往北京",
                             dt.datetime(2026, 10, 7, 12, 51, tzinfo=UTC), "https://www.cna.com.tw/x")
    assert out == post, (out, ch)


def test_an_unsupported_span_is_cut_whole():
    post = "Hegseth awarded the group.\n\nFrom February 28 to August 22, 2026, the group executed 3,693 sorties."
    out, ch = fix_time_words(post, "T\nHegseth announced on Wednesday the citation.",
                             dt.datetime(2026, 10, 7, 19, 49, tzinfo=UTC), "https://www.example.com/x")
    assert out == "Hegseth awarded the group.\n\nThe group executed 3,693 sorties.", (out, ch)


def test_no_source_no_change():
    post = "The S&P rose 0.6% on October 9."
    assert fix_caption_time_words(post, "T", "", dt.datetime(2026, 10, 9, 8, 5, tzinfo=UTC), "u") == (post, [])
    assert fix_caption_time_words("No comment", "T", LONG, None, "u") == ("No comment", [])
