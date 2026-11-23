import datetime as dt

import numpy as np
import pandas as pd
import pytest

from hitflop import features as F


@pytest.fixture(scope="module")
def rich(world):
    return F.add_research_features(world["lab"], world["merged"])


def test_holiday_dates_are_correct_for_known_festivals():
    major, _ = F._calendar(range(2015, 2027))
    known = [dt.date(2023, 11, 12), dt.date(2023, 3, 8), dt.date(2023, 4, 22), dt.date(2023, 10, 24), dt.date(2019, 10, 27), dt.date(2019, 6, 5), dt.date(2021, 11, 4), dt.date(2021, 10, 15)]
    for d in known:
        assert d in major, d


def test_calendar_flags_on_known_dates():
    c = F.calendar_features(pd.Series(pd.to_datetime(["2023-11-10", "2023-06-02", "2024-01-12"])))   # Fri before Diwali 12 Nov; ordinary Friday; Fri before Sankranti
    assert c.festival_window.tolist() == [1, 0, 1]
    assert c.is_friday.tolist() == [1, 1, 1]
    assert c.days_to_festival.iloc[0] == 2 and c.long_weekend.iloc[0] == 1


def test_wiki_rate_hand_computed():
    df = pd.DataFrame({"pv_90_31": [600.0], "pv_30_8": [230.0], "pv_7_1": [700.0]})
    o = F.wiki_rate_features(df)
    assert o.l_rate_early.iloc[0] == pytest.approx(np.log1p(10)) and o.l_rate_late.iloc[0] == pytest.approx(np.log1p(100))
    assert o.wiki_ratio_late_early.iloc[0] == pytest.approx(np.log(101 / 11))
    assert o.wiki_ratio_late_mid.iloc[0] == pytest.approx(np.log(101 / 11))


def test_yt_ratio_hand_computed():
    df = pd.DataFrame({"cm_pre_likes": [100.0], "cm_pre_n": [50.0], "campaign_lead_days": [10.0], "n_promos": [5.0], "trailer_lead_days": [5.0]})
    o = F.yt_ratio_features(df)
    assert o.yt_likes_per_comment.iloc[0] == pytest.approx(np.log1p(2)) and o.yt_comments_per_campaign_day.iloc[0] == pytest.approx(np.log1p(5))
    assert o.yt_trailer_share_of_campaign.iloc[0] == pytest.approx(0.5)


def test_competition_counts_hand_computed():
    allm = pd.DataFrame({"tmdb_id": [1, 2, 3, 4, 5], "language": ["hi", "hi", "ta", "hi", "hi"],
                         "release_date": pd.to_datetime(["2020-01-10", "2020-01-12", "2020-01-14", "2020-01-20", "2020-03-01"])})
    o = F.competition_features(allm.iloc[[0]], allm)
    assert o.iloc[0].tolist() == [2, 1, 2]        # +-7d: ids 2,3 (hi,ta); same-language +-7d: id 2; +-14d same language: ids 2,4


def test_competition_excludes_self(rich):
    assert (rich.n_open_pm7_all >= 0).all() and rich.n_open_pm14_same_lang.max() < 80


def test_features_do_not_use_outcomes(world):
    """Scramble revenue, roi and hit: no research feature may change."""
    lab = world["lab"].copy()
    base = F.add_research_features(lab, world["merged"])[F.ALL_RESEARCH]
    rng = np.random.default_rng(1)
    lab2 = lab.copy()
    for c in ("revenue", "roi", "hit"):
        lab2[c] = rng.permutation(lab2[c].values)
    m2 = world["merged"].copy()
    m2["revenue"] = rng.permutation(m2["revenue"].values)
    alt = F.add_research_features(lab2, m2)[F.ALL_RESEARCH]
    pd.testing.assert_frame_equal(base, alt)


def test_shapes_and_finite(rich):
    assert len(rich) == 854
    for cols in F.RESEARCH_GROUPS.values():
        assert set(cols) <= set(rich.columns)
    assert np.isfinite(rich[F.CALENDAR + F.COMPETITION].values).all()
    w = rich[F.WIKI_RATE]
    assert w.notna().sum().min() == 700 and np.isfinite(w.dropna().values).all()
