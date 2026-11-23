"""Upgrade Phase 4b: out-of-distribution detection and uncertainty warnings.

A `reference` is a plain dictionary built from the films a model was trained on. It holds
  * a Mahalanobis-distance model of the film's numeric inputs (budget, runtime, release month and, for the buzz model, the three page-view windows),
    with a shrunk covariance, and a distance limit set BEFORE looking at any outcome: the 97.5th percentile of the reference films' own distances;
  * the range of budget and runtime (extreme range = outside what was seen; usual range = 1st to 99th percentile);
  * how many reference films share each language, and which genres were ever seen.
`assess` turns a film's inputs and the model's probability into a level (normal, caution, high) and a list of plain-language reasons.
Reasons are about the INPUTS and the decision margin; none of them uses the label, so they can be shown before release.
"""
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

BASIC_COLS = ["log_budget", "runtime", "month_sin", "month_cos"]
BUZZ_COLS = BASIC_COLS + ["l_pv_90_31", "l_pv_30_8", "l_pv_7_1"]
LIMIT_PERCENTILE = 97.5
MIN_LANGUAGE_FILMS = 30
CLOSE_CALL_MARGIN = 0.05           # chosen before any evaluation: half of the 0.10-wide band around the Hit line


def _genres(series):
    return {g.strip() for t in series.fillna("") for g in str(t).split(",") if g.strip()}


def fit_reference(df, mode="basic"):
    """Reference dictionary from the training films `df` (a feature table such as `world.build_world()[2]`)."""
    cols = BUZZ_COLS if mode == "buzz" else BASIC_COLS
    X = df[cols].astype(float)
    med = X.median()
    Xf = X.fillna(med)
    mu, sd = Xf.mean().values, Xf.std(ddof=0).replace(0, 1).values
    Z = (Xf.values - mu) / sd
    lw = LedoitWolf().fit(Z)
    d2 = np.einsum("ij,jk,ik->i", Z, lw.precision_, Z) / len(cols)
    return {"mode": mode, "cols": cols, "median": med.to_dict(), "mu": mu, "sd": sd, "precision": lw.precision_, "limit": float(np.percentile(d2, LIMIT_PERCENTILE)),
            "range": {c: [float(df[c].min()), float(df[c].quantile(.01)), float(df[c].quantile(.99)), float(df[c].max())] for c in ("log_budget", "runtime")},
            "language_counts": df.language.value_counts().to_dict(), "genres": sorted(_genres(df.genres)), "n": int(len(df))}


def distance(ref, df):
    """Normalised squared Mahalanobis distance of each row (about 1 for a typical film)."""
    X = df[ref["cols"]].astype(float).fillna(pd.Series(ref["median"]))
    Z = (X.values - ref["mu"]) / ref["sd"]
    return np.einsum("ij,jk,ik->i", Z, ref["precision"], Z) / len(ref["cols"])


def flags(ref, df):
    """Boolean table of input warnings, one row per film in `df`."""
    out = pd.DataFrame(index=df.index)
    out["mahalanobis_high"] = distance(ref, df) > ref["limit"]
    for c, name in (("log_budget", "budget"), ("runtime", "runtime")):
        mn, q1, q99, mx = ref["range"][c]
        out[f"{name}_outside_seen"] = (df[c] < mn) | (df[c] > mx)
        out[f"{name}_unusual"] = ((df[c] < q1) | (df[c] > q99)) & ~out[f"{name}_outside_seen"]
    out["language_few_films"] = df.language.map(lambda l: ref["language_counts"].get(l, 0) < MIN_LANGUAGE_FILMS)
    seen = set(ref["genres"])
    out["unseen_genre"] = df.genres.map(lambda t: bool(_genres(pd.Series([t])) - seen))
    out["input_shift"] = out[["mahalanobis_high", "budget_outside_seen", "runtime_outside_seen"]].any(axis=1)
    out["any_input_flag"] = out.drop(columns="input_shift").any(axis=1)
    return out


def assess(ref, row, probability, threshold):
    """`row` is a one-row feature DataFrame. Returns {'level', 'reasons', 'notes', 'distance', 'distance_limit', 'close_call'}.

    Levels follow what the development evidence supports (see docs/UPGRADE_LOG.md, Phase 4):
      high     a budget or runtime outside everything the model was trained on: the model is extrapolating (true by construction)
      caution  close call, within 5 points of the Hit line: on development films the deployed models were right only about 40% of the time there
      note     the film looks unusual (input mix, extreme but seen values, few films in the language, unseen genre). On past films such films were NOT predicted worse,
               so this is information about the training data, not a reliability alarm
      normal   nothing to report
    `reasons` holds the high and caution messages; `notes` holds the note messages."""
    f = flags(ref, row).iloc[0]
    d = float(distance(ref, row)[0])
    close = abs(probability - threshold) <= CLOSE_CALL_MARGIN
    high, caution, notes = [], [], []
    if f.budget_outside_seen:
        high.append("The budget is outside every budget the model was trained on, so the estimate is an extrapolation.")
    if f.runtime_outside_seen:
        high.append("The runtime is outside every runtime the model was trained on, so the estimate is an extrapolation.")
    if close:
        caution.append(f"Close call: the chance is within {CLOSE_CALL_MARGIN * 100:.0f} points of the Hit line. On past films the verdict was right only about 40% of the time this close to the line, so treat it as a toss-up.")
    if f.mahalanobis_high:
        notes.append("This film's mix of budget, runtime and release month is unusual compared with the training films.")
    if f.budget_unusual:
        notes.append("The budget is at the extreme end of the training films (outside their 1st to 99th percentile range).")
    if f.runtime_unusual:
        notes.append("The runtime is at the extreme end of the training films.")
    if f.language_few_films:
        notes.append(f"Few training films are in this language (fewer than {MIN_LANGUAGE_FILMS}), so the model has little to learn from.")
    if f.unseen_genre:
        notes.append("At least one chosen genre never appeared in the training films.")
    level = "high" if high else "caution" if caution else "note" if notes else "normal"
    return {"level": level, "reasons": high + caution, "notes": notes, "distance": d, "distance_limit": ref["limit"], "close_call": bool(close)}


def attach_references(path=None):
    """Add `uncertainty` references (built from all labelled films, the films the deployed models were trained on) to an existing artifact without refitting any model."""
    import joblib

    from . import artifacts as A
    from .world import build_world
    path = path or A.PATH
    art = joblib.load(path)
    _, lab, _, _ = build_world()
    art["uncertainty"] = {m: fit_reference(lab, m) for m in ("basic", "buzz")}
    joblib.dump(art, path, compress=3)
    return {m: (r["n"], round(r["limit"], 3)) for m, r in art["uncertainty"].items()}


if __name__ == "__main__":
    print(attach_references())
