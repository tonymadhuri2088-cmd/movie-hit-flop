"""Pure functions behind the dashboard: what-if curves, side-by-side film comparison, leaderboard tables. No Streamlit here, so everything is testable."""
import numpy as np
import pandas as pd

from .. import config as C
from .. import data
from .. import explain as X
from .. import predict as P
from .. import uncertainty as UC

LANG_NAMES = {"hi": "Hindi", "ta": "Tamil", "te": "Telugu", "ml": "Malayalam", "kn": "Kannada"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ------------------------------------------------------------------ what-if curves
def _probability(art, mode, features):
    a = art[mode]
    return float(a["pipeline"].predict_proba(features[a["columns"]])[:, 1][0])


def _with(features, **changes):
    """The model row with some raw inputs replaced, re-deriving the deterministic row-only features (log budget, month terms, season)."""
    f = features.copy()
    for k, v in changes.items():
        f[k] = v
    f["release_date"] = pd.to_datetime(f["release_date"])
    return data.build_basic_features(f)


def what_if(art, features, mode):
    """Hit chance as one input at a time is varied and everything else stays as entered. The model is only a pattern-finder, so these are associations, not advice."""
    base = float(features["budget"].iloc[0])
    rng = ((art.get("uncertainty") or {}).get(mode) or {}).get("range", {})
    lo_b, hi_b = (10 ** rng["log_budget"][1], 10 ** rng["log_budget"][2]) if rng else (1e6, 1e8)
    budgets = np.unique(np.round(np.geomspace(max(C.MIN_BUDGET * 1.0001, lo_b / 3), hi_b * 3, 15), -3))
    out = {}
    out["budget"] = pd.DataFrame({"x": budgets / 1e6, "label": [f"${b / 1e6:.1f}M" for b in budgets],
                                  "probability": [_probability(art, mode, _with(features, budget=b)) for b in budgets], "in_training_range": (budgets >= lo_b) & (budgets <= hi_b)})
    rts = np.arange(70, 201, 10)
    rr = rng.get("runtime") if rng else None
    out["runtime"] = pd.DataFrame({"x": rts, "label": [f"{r} min" for r in rts], "probability": [_probability(art, mode, _with(features, runtime=float(r))) for r in rts],
                                   "in_training_range": [(rr is None) or (rr[1] <= r <= rr[2]) for r in rts]})
    langs = list(LANG_NAMES)
    out["language"] = pd.DataFrame({"x": langs, "label": [LANG_NAMES[g] for g in langs], "probability": [_probability(art, mode, _with(features, language=g)) for g in langs],
                                    "in_training_range": [((art.get("uncertainty") or {}).get(mode) or {}).get("language_counts", {}).get(g, 99) >= UC.MIN_LANGUAGE_FILMS for g in langs]})
    rd = pd.Timestamp(features["release_date"].iloc[0])
    dates = [pd.Timestamp(year=rd.year, month=m, day=15) for m in range(1, 13)]
    out["month"] = pd.DataFrame({"x": list(range(1, 13)), "label": MONTHS, "probability": [_probability(art, mode, _with(features, release_date=d)) for d in dates], "in_training_range": True})
    out["current"] = {"budget": base / 1e6, "runtime": float(features["runtime"].iloc[0]), "language": str(features["language"].iloc[0]), "month": int(rd.month)}
    return out


# ------------------------------------------------------------------ side-by-side comparison
def compare_films(art, films, mode, top=5):
    """Predict several films and list the strongest reasons for each. `films` is a list of movie dictionaries (as in `predict.predict`).
    A film that cannot be scored is reported with an `error` instead of stopping the others."""
    results = []
    for movie in films:
        try:
            out = P.predict(art, movie, mode)
        except P.InputError as e:
            results.append({"name": movie.get("name", "film"), "error": str(e)})
            continue
        a = art[mode]
        f = out["features"]
        vals, Z, names = X.shap_values(a["pipeline"], a["shap_background"], f[a["columns"]])
        le = X.local_explanation(vals[0], Z[0], names, top=top)
        unc = (art.get("uncertainty") or {}).get(mode)
        assess = UC.assess(unc, f, out["probability"], a["threshold"]) if unc else None
        results.append({"name": movie.get("name", "film"), "probability": out["probability"], "label": out["label"], "threshold": a["threshold"],
                        "drivers": le, "level": assess["level"] if assess else "normal", "reasons": (assess["reasons"] if assess else []), "movie": movie})
    return results


def comparison_frame(results):
    ok = [r for r in results if "error" not in r]
    return pd.DataFrame([{"film": r["name"], "chance_of_hit": r["probability"], "verdict": r["label"], "hit_line": r["threshold"], "margin_to_line": r["probability"] - r["threshold"],
                          "reliability": {"normal": "no warnings", "note": "note", "caution": "caution", "high": "treat with care"}.get(r["level"], r["level"])} for r in ok])


# ------------------------------------------------------------------ leaderboard
def leaderboard(lb, metric="validation AUC"):
    """19-model table ordered by `metric`, with a plain flag for 'its test interval overlaps the best model's interval' (informal; the formal comparison is in the research lab)."""
    d = lb.copy().sort_values(metric, ascending=False).reset_index(drop=True)
    d.insert(0, "rank", np.arange(1, len(d) + 1))
    best_by_test = d.sort_values("test AUC", ascending=False).iloc[0]
    d["interval_overlaps_best_test"] = d.ci_high >= best_by_test.ci_low
    d["validation_to_test_gap"] = d["test AUC"] - d["validation AUC"]
    return d


def tied_share(d):
    """Share of models whose test interval overlaps the best test model's interval."""
    return float(d.interval_overlaps_best_test.mean())
