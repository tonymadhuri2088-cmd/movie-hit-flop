"""Phase 2 feature engineering. Every feature here is computable on the day before release from information that exists by then.

Groups (all optional, evaluated one at a time in Phase 3 so each can be accepted or rejected on evidence):
  WIKI_RATE    Wikipedia page views as per-day rates over the three pre-release windows, and ratios between them (is attention rising?)
  YT_RATIO     YouTube campaign intensity and engagement ratios
  CALENDAR     day of week, Indian festival and long-weekend flags (fixed calendars, known years ahead)
  COMPETITION  how many other films (file-wide, any outcome) open within +-7/14 days. Uses only other films' release dates, never their outcomes or buzz.
"""
import datetime as dt

import holidays
import numpy as np
import pandas as pd

# Wikipedia windows end the day before release: [-90,-31] = 60 days, [-30,-8] = 23 days, [-7,-1] = 7 days.
WIN_DAYS = {"early": 60, "mid": 23, "late": 7}
WIKI_RATE = ["l_rate_early", "l_rate_mid", "l_rate_late", "wiki_ratio_late_early", "wiki_ratio_mid_early", "wiki_ratio_late_mid"]
YT_RATIO = ["yt_likes_per_comment", "yt_comments_per_campaign_day", "yt_promos_per_campaign_day", "yt_trailer_share_of_campaign"]
CALENDAR = ["release_dow", "is_friday", "festival_window", "days_to_festival", "holidays_in_7d", "long_weekend"]
COMPETITION = ["n_open_pm7_all", "n_open_pm7_same_lang", "n_open_pm14_same_lang"]
RESEARCH_GROUPS = {"wiki_rate": WIKI_RATE, "yt_ratio": YT_RATIO, "calendar": CALENDAR, "competition": COMPETITION}
ALL_RESEARCH = sum(RESEARCH_GROUPS.values(), [])

# Festivals that move film release strategy. Library dates for Diwali/Holi/Eid/Dussehra were spot-checked in tests; Pongal/Sankranti is a fixed solar date.
MAJOR = ("Diwali", "Holi", "Id-ul-Fitr", "Id-ul-Zuha", "Dussehra", "Christmas", "Independence Day", "Republic Day")


def _calendar(years):
    h = holidays.India(years=years)
    major = sorted(d for d, n in h.items() if any(k in n for k in MAJOR))
    major += [dt.date(y, 1, 14) for y in years]            # Makar Sankranti / Pongal (fixed solar date, matters for Telugu and Tamil releases)
    return sorted(set(major)), sorted(h.keys())


def calendar_features(release_dates):
    rd = pd.to_datetime(pd.Series(release_dates))
    years = list(range(int(rd.dt.year.min()) - 1, int(rd.dt.year.max()) + 2))
    major, anyhol = _calendar(years)
    major_a, any_a = np.array(major, dtype="datetime64[D]"), np.array(anyhol, dtype="datetime64[D]")
    out = pd.DataFrame(index=rd.index)
    d = rd.values.astype("datetime64[D]")
    out["release_dow"] = rd.dt.dayofweek.values
    out["is_friday"] = (out.release_dow == 4).astype(int)
    diff = (major_a[None, :] - d[:, None]).astype(int)                                 # festival minus release, in days
    out["festival_window"] = ((diff >= -2) & (diff <= 4)).any(axis=1).astype(int)       # a festival from 2 days before to 4 days after opening
    out["days_to_festival"] = np.minimum(np.abs(diff).min(axis=1), 30)
    diff_all = (any_a[None, :] - d[:, None]).astype(int)
    out["holidays_in_7d"] = ((diff_all >= 0) & (diff_all <= 6)).sum(axis=1)
    out["long_weekend"] = (out.release_dow.isin([3, 4]).values & ((diff_all >= 0) & (diff_all <= 4)).any(axis=1)).astype(int)
    return out


def wiki_rate_features(df):
    out = pd.DataFrame(index=df.index)
    r = {k: df[c] / WIN_DAYS[k] for k, c in (("early", "pv_90_31"), ("mid", "pv_30_8"), ("late", "pv_7_1"))}
    for k in r:
        out[f"l_rate_{k}"] = np.log1p(r[k])
    out["wiki_ratio_late_early"] = np.log((r["late"] + 1) / (r["early"] + 1))
    out["wiki_ratio_mid_early"] = np.log((r["mid"] + 1) / (r["early"] + 1))
    out["wiki_ratio_late_mid"] = np.log((r["late"] + 1) / (r["mid"] + 1))
    return out


def yt_ratio_features(df):
    camp = df.campaign_lead_days.clip(lower=1, upper=365)
    out = pd.DataFrame(index=df.index)
    out["yt_likes_per_comment"] = np.log1p(df.cm_pre_likes / df.cm_pre_n.clip(lower=1))
    out["yt_comments_per_campaign_day"] = np.log1p(df.cm_pre_n / camp)
    out["yt_promos_per_campaign_day"] = np.log1p(df.n_promos / camp)
    out["yt_trailer_share_of_campaign"] = (df.trailer_lead_days.clip(lower=0, upper=365) / camp).clip(upper=1)
    return out


def competition_features(df, all_movies):
    """Counts of OTHER films in the file opening within +-7 / +-14 days (any outcome status). Only release dates and languages are used."""
    a = all_movies.dropna(subset=["release_date"])
    dates = a.release_date.values.astype("datetime64[D]")
    langs = a.language.values
    ids = a.tmdb_id.values
    out = pd.DataFrame(index=df.index, columns=COMPETITION, dtype=float)
    for i, (t, lg, tid) in enumerate(zip(df.release_date.values.astype("datetime64[D]"), df.language.values, df.tmdb_id.values)):
        gap = np.abs((dates - t).astype(int))
        other = ids != tid
        w7, w14 = (gap <= 7) & other, (gap <= 14) & other
        out.iloc[i] = [w7.sum(), (w7 & (langs == lg)).sum(), (w14 & (langs == lg)).sum()]
    return out.astype(float)


def add_research_features(lab, all_movies):
    """Return `lab` with all Phase 2 columns added. `lab` must contain the columns of the original merged data."""
    parts = [wiki_rate_features(lab), yt_ratio_features(lab), calendar_features(lab.release_date), competition_features(lab, all_movies)]
    drop = [c for c in ALL_RESEARCH if c in lab.columns]
    return pd.concat([lab.drop(columns=drop)] + parts, axis=1)
