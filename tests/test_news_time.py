"""core/news_time.py on cases from AM1ST's own posts of 10/06-10/10."""
from datetime import date, datetime, timezone

from core.news_time import decide, first_disclosure, latest_moment, names_this_outlet


def utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


def dev(what, when=None, kind="said", cited=None):
    return {"what": what, "when": when, "kind": kind, "about_when": None, "cited_outlet": cited}


def test_plans_and_vague_times_are_not_evidence():
    oct8 = date(2026, 10, 8)
    assert latest_moment("Tuesday, Oct. 13", oct8) is None          # the Wasilla rally, not last year's
    assert latest_moment("will be heard Friday", oct8) is None
    assert latest_moment("last week", oct8) is None                 # Emmer / Axios
    assert latest_moment("in August", oct8) is None
    assert latest_moment("Oct. 4", oct8).isoformat().startswith("2026-10-04T23:59")
    assert latest_moment("Monday", oct8).date() == date(2026, 10, 5)
    assert latest_moment("Dec. 20", date(2026, 1, 5)).year == 2025


def test_this_outlet_and_no_other():
    assert names_this_outlet("Blaze News", "https://www.theblaze.com/news/x")
    assert names_this_outlet("The Post", "https://nypost.com/2026/10/07/x")
    assert names_this_outlet("Fox News Digital", "https://www.foxnews.com/politics/x")
    assert not names_this_outlet("California Post", "https://californiaglobe.com/x")


def test_first_reports():
    assert first_disclosure("x", "U.S. Customs and Border Protection told Blaze News the incident unfolded on Sept. 29.",
                            "https://www.theblaze.com/news/meth")
    assert first_disclosure("Dem fundraising giant", "FIRST ON FOX: Democrats' largest platform says", "https://www.foxnews.com/politics/x")
    assert first_disclosure("Scoop: Senate deal near", "", "https://www.axios.com/x")
    assert first_disclosure("x", "When Blaze News visited office buildings in Portland", "https://www.theblaze.com/news/maine")
    assert not first_disclosure("x", "New videos obtained exclusively by CBS News show", "https://revolver.news/x")
    assert not first_disclosure("x", "an official told the California Post", "https://californiaglobe.com/x")


NOW = utc(2026, 10, 7, 13, 47)
PUB = utc(2026, 10, 7, 4, 0)


def test_relay_of_an_interview_three_days_on_is_stale():
    body = ("Thom Tillis brazenly asserted that the next Republican presidential nominee cannot be a MAGA candidate.\n"
            "CBS aired its Tillis interview on October 4 as the senator promotes his new book.\n"
            "As The Gateway Pundit reported Tuesday, Tillis appeared on The View.")
    devs = [dev("Thom Tillis brazenly asserted that the next Republican presidential nominee cannot be a MAGA candidate.", "October 4", cited="CBS"),
            dev("Tillis appeared on The View.", "Tuesday", kind="happened")]
    r = decide(devs, "Tillis Runs to 60 Minutes", body, "https://www.thegatewaypundit.com/2026/10/tillis/", PUB, NOW)
    assert r["verdict"] == "stale", r


def test_old_event_newly_reported_elsewhere_is_fresh():
    body = ("But on September 17, it turned into a firefight. According to a Reuters exclusive published Friday, "
            "a team working for Prince's Vectus Global was ambushed in South Kivu.")
    devs = [dev("a team working for Prince's Vectus Global was ambushed in South Kivu", "September 17", kind="happened", cited="Reuters")]
    r = decide(devs, "Blackwater Bloodied", body, "https://www.zerohedge.com/geopolitical/x",
               utc(2026, 10, 10, 1, 0), utc(2026, 10, 10, 1, 37))
    assert r["verdict"] == "fresh", r


def test_video_posted_this_week_is_fresh():
    body = ("At the Georgia State Election Board meeting on September 28, a lawyer told the board that Fulton County did nothing wrong.\n"
            "Her argument, captured by VoterGA and posted this week by Garland Favorito, was that the NVRA creates a special category.")
    devs = [dev("At the Georgia State Election Board meeting on September 28, a lawyer told the board that Fulton County did nothing wrong.", "September 28")]
    r = decide(devs, "88-Year-Old Fulton Voter", body, "https://www.thegatewaypundit.com/2026/10/fulton/",
               utc(2026, 10, 8, 14, 0), utc(2026, 10, 8, 18, 21))
    assert r["verdict"] == "fresh", r


def test_new_remarks_under_an_old_event_are_fresh():
    # The model headed Joe Abraham's 10/09 remarks with his daughter's crash.
    body = ("Katie was with her friend on Jan. 3 when a drunk driver hit them.\n"
            "On October 9, 2026, Abraham condemned that view.")
    devs = [dev("Katie was with her friend on Jan. 3 when a drunk driver hit them.", "Jan. 3", kind="happened"),
            dev("On October 9, 2026, Abraham condemned that view.", "October 9, 2026")]
    url, pub, now = "https://pjmedia.com/x", utc(2026, 10, 10, 6, 0), utc(2026, 10, 10, 8, 24)
    assert decide(devs, "Angel Dad Explains", body, url, pub, now)["verdict"] == "fresh"
    # a new thing that merely happened does not count (Tillis on The View)
    devs[1]["kind"] = "happened"
    assert decide(devs, "Angel Dad Explains", body, url, pub, now)["verdict"] == "stale"


def test_made_up_sentence_or_time_is_not_evidence():
    body = "Signed by Democrat Gov. Gavin Newsom on September 27, AB 1803 amends current law."
    real = dev("Signed by Democrat Gov. Gavin Newsom on September 27, AB 1803 amends current law.", "September 27", kind="happened")
    now, pub = utc(2026, 10, 8, 15, 31), utc(2026, 10, 8, 12, 0)
    assert decide([real], "New California law", body, "https://www.lifesitenews.com/x", pub, now)["verdict"] == "stale"
    invented = dict(real, what="Newsom signed AB 1803 last month.")
    assert decide([invented], "New California law", body, "https://www.lifesitenews.com/x", pub, now)["verdict"] == "fresh"
    wrong_time = dict(real, when="Sept. 2")
    assert decide([wrong_time], "New California law", body, "https://www.lifesitenews.com/x", pub, now)["verdict"] == "fresh"


def test_inside_the_window_is_fresh_and_day_end_is_generous():
    body = "Customs and Border Protection says it has verified more than 1 billion travelers, in an Oct. 5 statement."
    d = [dev("Customs and Border Protection says it has verified more than 1 billion travelers, in an Oct. 5 statement.", "Oct. 5")]
    url = "https://www.zerohedge.com/technology/x"
    # Oct. 5 counts from 23:59 Eastern = Oct. 6 03:59 UTC; 48h later is Oct. 8 03:59 UTC
    assert decide(d, "CBP", body, url, utc(2026, 10, 7, 12, 0), utc(2026, 10, 8, 3, 0))["verdict"] == "fresh"
    assert decide(d, "CBP", body, url, utc(2026, 10, 7, 12, 0), utc(2026, 10, 8, 23, 4))["verdict"] == "stale"


def test_first_report_is_fresh_without_a_model():
    r = decide([], "Exclusive | JD Vance's tiny hometown is overrun", "text", "https://nypost.com/x",
               utc(2026, 10, 7, 16, 0), utc(2026, 10, 7, 18, 10))
    assert r["verdict"] == "fresh" and "first report" in r["reason"]


def test_month_year_ranges_and_closed_doors():
    oct10 = date(2026, 10, 10)
    assert latest_moment("September 2026", oct10).date() == date(2026, 9, 30)   # not Sept. 20
    assert latest_moment("in August 2023", oct10).date() == date(2023, 8, 31)
    assert latest_moment("from October 4 to 8", oct10).date() == date(2026, 10, 8)
    assert latest_moment("Sept. 30 to Oct. 2", oct10).date() == date(2026, 10, 2)
    body = ("The new details were laid out by Pentagon officials on Sept. 29 in a roughly hour-long closed-door "
            "meeting with House staff.")
    d = [dev(body, "Sept. 29", cited="Bloomberg")]
    r = decide(d, "F-35 parts", body, "https://www.zerohedge.com/technology/x", utc(2026, 10, 8, 1, 0), utc(2026, 10, 8, 1, 50))
    assert r["verdict"] == "fresh", r


def test_data_cutoffs_and_points_of_comparison_are_not_news_times():
    oct10 = date(2026, 10, 10)
    assert latest_moment("as of June 30, 2026", oct10) is None
    assert latest_moment("since Oct. 2", oct10) is None
    body = "Gold rose above $4,200 for the first time since Oct. 2, up 1.6% on the day."
    d = [dev(body, "Oct. 2", kind="happened")]
    r = decide(d, "Gold tops $4,200", body, "https://www.example.com/x", utc(2026, 10, 9, 5, 0), utc(2026, 10, 9, 6, 8))
    assert r["verdict"] == "fresh", r


def test_plans_own_interviews_and_undated_relays():
    # a plan with a bare weekday is the coming one, not last week's
    body = "EU trade chief Maros Sefcovic will head to Beijing on Thursday for two days of meetings."
    d = [dev(body, "Thursday")]
    assert decide(d, "EU, China to hold talks", body, "https://www.example.com/x",
                  utc(2026, 10, 5, 1, 0), utc(2026, 10, 5, 2, 40))["verdict"] == "fresh"
    # a plan whose full date has passed: the piece was written before its event
    body = "The case will be heard by the Supreme Court on Oct. 5."
    d = [dev(body, "Oct. 5", kind="happened")]
    assert decide(d, "SCOTUS", body, "https://www.westernjournal.com/x",
                  utc(2026, 10, 10, 10, 0), utc(2026, 10, 10, 12, 37))["verdict"] == "stale"
    # the outlet's own interview
    assert first_disclosure("VOA专访俞大㵢：美对台政策没有改变", "", "https://www.voachinese.com/a/x")
    # something that happened, known through someone's undated report
    body = "Shi Tingfu, 67, died on Sept. 22 in prison, according to the Washington-based nonprofit."
    d = [dev(body, "Sept. 22", kind="happened", cited="Dui Hua Foundation")]
    assert decide(d, "Activist dies in prison", body, "https://www.example.com/x",
                  utc(2026, 10, 6, 10, 0), utc(2026, 10, 6, 13, 26))["verdict"] == "fresh"
