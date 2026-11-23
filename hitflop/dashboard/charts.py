"""Altair charts in the dark house style. Every function returns a styled chart; none reads files."""
import altair as alt
import numpy as np
import pandas as pd

from . import theme as T


def _style(c, height=None):
    return T.altair_style(c, height)


def auc_forest(df, name, auc, lo, hi, domain=(0.45, 0.9), xtitle="AUC (dot) with 95% interval; 0.5 = guessing", height=None, color=None):
    d = df[[name, auc, lo, hi]].copy()
    d.columns = ["model", "auc", "lo", "hi"]
    order = d.sort_values("auc", ascending=False)["model"].tolist()
    y = alt.Y("model:N", sort=order, title=None, axis=alt.Axis(labelLimit=300, ticks=False, domain=False))
    xs = alt.Scale(domain=list(domain))
    bar = alt.Chart(d).mark_rule(color=T.SLATE, strokeWidth=3).encode(y=y, x=alt.X("lo:Q", scale=xs, title=xtitle), x2="hi:Q")
    dot = alt.Chart(d).mark_circle(size=120, color=color or T.TEXT).encode(y=y, x="auc:Q", tooltip=["model", alt.Tooltip("auc:Q", format=".3f"), alt.Tooltip("lo:Q", format=".3f"), alt.Tooltip("hi:Q", format=".3f")])
    chance = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(color=T.FLOP, strokeDash=[4, 4]).encode(x=alt.X("x:Q", scale=xs))
    return _style(chance + bar + dot, height or alt.Step(30))


def scatter_val_test(lb):
    d = lb.copy()
    d["label"] = d["model"]
    lo, hi = float(min(d["validation AUC"].min(), d["test AUC"].min())) - 0.02, float(max(d["validation AUC"].max(), d["test AUC"].max())) + 0.02
    sc = alt.Scale(domain=[lo, hi])
    pts = alt.Chart(d).mark_circle(size=90, color=T.GOLD, opacity=.9).encode(
        x=alt.X("validation AUC:Q", scale=sc, title="Validation AUC (chronological folds, used to choose)"),
        y=alt.Y("test AUC:Q", scale=sc, title="Test AUC (frozen 2023+ films, never used to choose)"),
        tooltip=["model", "feature group", alt.Tooltip("validation AUC:Q", format=".3f"), alt.Tooltip("test AUC:Q", format=".3f")])
    diag = alt.Chart(pd.DataFrame({"a": [lo, hi], "b": [lo, hi]})).mark_line(color=T.SLATE, strokeDash=[4, 4]).encode(x=alt.X("a:Q", scale=sc), y=alt.Y("b:Q", scale=sc))
    return _style(diag + pts, 360)


def what_if_chart(df, kind, current, threshold, xtitle):
    d = df.copy()
    d["range"] = np.where(d["in_training_range"], "typical of the training films", "outside what the model saw")
    cols = alt.Scale(domain=["typical of the training films", "outside what the model saw"], range=[T.TEXT, T.SLATE])
    ty = alt.Y("probability:Q", scale=alt.Scale(domain=[0, 1]), title="Chance of a Hit", axis=alt.Axis(format="%"))
    line = alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(color=T.GOLD, strokeWidth=2).encode(y="y:Q")
    tip = [alt.Tooltip("label:N", title=xtitle), alt.Tooltip("probability:Q", format=".0%", title="Chance of a Hit")]
    if kind == "language":
        base = alt.Chart(d).mark_bar(cornerRadius=3).encode(x=alt.X("label:N", sort=None, title=xtitle, axis=alt.Axis(labelAngle=0)), y=ty,
                                                            color=alt.Color("range:N", scale=cols, legend=alt.Legend(title=None)), tooltip=tip)
        cur = alt.Chart(d[d.x == current]).mark_bar(cornerRadius=3, stroke=T.GOLD, strokeWidth=3, fillOpacity=0).encode(x=alt.X("label:N", sort=None), y=ty)
        return _style(base + cur + line, 260)
    xs = alt.X("x:Q", title=xtitle, scale=alt.Scale(type="log") if kind == "budget" else alt.Scale(zero=False))
    if kind == "month":
        xs = alt.X("label:N", sort=list(d.label), title=xtitle, axis=alt.Axis(labelAngle=0))
    base = alt.Chart(d).mark_line(color=T.SLATE, strokeWidth=2).encode(x=xs, y=ty)
    pts = alt.Chart(d).mark_circle(size=80).encode(x=xs, y=ty, color=alt.Color("range:N", scale=cols, legend=alt.Legend(title=None)), tooltip=tip)
    if kind == "month":
        cur_d = d[d.x == current]
    else:
        cur_d = pd.DataFrame({"x": [current]})
    mark = alt.Chart(cur_d).mark_rule(color=T.GOLD, strokeDash=[2, 3]).encode(x=(alt.X("label:N", sort=list(d.label)) if kind == "month" else alt.X("x:Q", scale=alt.Scale(type="log") if kind == "budget" else alt.Scale(zero=False))))
    return _style(base + pts + mark + line, 260)


def driver_bars(compare_results):
    rows = []
    for r in compare_results:
        if "error" in r:
            continue
        d = r["drivers"].copy()
        d["film"] = r["name"]
        rows.append(d)
    if not rows:
        return None
    d = pd.concat(rows)
    return d


def probability_bars(frame):
    d = frame.copy()
    d["colour"] = np.where(d.verdict == "Hit", "Hit", "Flop")
    order = list(d.film)
    base = alt.Chart(d).mark_bar(cornerRadius=3, height=26).encode(
        y=alt.Y("film:N", sort=order, title=None, axis=alt.Axis(ticks=False, domain=False, labelLimit=240)),
        x=alt.X("chance_of_hit:Q", scale=alt.Scale(domain=[0, 1]), title="Chance of earning at least 2x the budget", axis=alt.Axis(format="%")),
        color=alt.Color("colour:N", scale=alt.Scale(domain=["Hit", "Flop"], range=[T.HIT, T.FLOP]), legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("film:N"), alt.Tooltip("chance_of_hit:Q", format=".0%"), alt.Tooltip("verdict:N")])
    line = alt.Chart(d).mark_tick(color=T.GOLD, thickness=3, size=34).encode(y=alt.Y("film:N", sort=order), x="hit_line:Q")
    return _style(base + line, alt.Step(46))


def ledger_stack(summ):
    d = summ.copy()
    d["name"] = d.phase + ": " + d.experiment
    long = d.melt(id_vars=["name"], value_vars=["supported_better", "no_clear_difference", "supported_worse"], var_name="result", value_name="count")
    long["result"] = long.result.map({"supported_better": "better (supported)", "no_clear_difference": "no clear difference", "supported_worse": "worse (supported)"})
    order = list(d.name)
    return _style(alt.Chart(long).mark_bar().encode(
        y=alt.Y("name:N", sort=order, title=None, axis=alt.Axis(labelLimit=420, ticks=False, domain=False)), x=alt.X("count:Q", title="Number of comparisons"),
        color=alt.Color("result:N", scale=alt.Scale(domain=["better (supported)", "no clear difference", "worse (supported)"], range=[T.HIT, "#3a4650", T.FLOP]), legend=alt.Legend(title=None)),
        tooltip=["name", "result", "count"]), max(220, 34 * len(d)))


def delta_forest(df, label, diff, lo, hi, verdict=None, height=None, xtitle="Change in AUC (95% interval)"):
    d = df[[label, diff, lo, hi] + ([verdict] if verdict else [])].copy()
    d.columns = ["label", "diff", "lo", "hi"] + (["verdict"] if verdict else [])
    order = list(d.label)
    y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=300, ticks=False, domain=False))
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=T.SLATE).encode(x="x:Q")
    rule = alt.Chart(d).mark_rule(strokeWidth=3, color=T.SLATE).encode(y=y, x=alt.X("lo:Q", title=xtitle), x2="hi:Q")
    col = alt.Color("verdict:N", scale=alt.Scale(domain=["better (supported)", "no clear difference", "worse (supported)"], range=[T.HIT, T.TEXT, T.FLOP]), legend=alt.Legend(title=None)) if verdict else alt.value(T.TEXT)
    dot = alt.Chart(d).mark_circle(size=90).encode(y=y, x="diff:Q", color=col, tooltip=["label", alt.Tooltip("diff:Q", format="+.3f")])
    return _style(zero + rule + dot, height or alt.Step(30))


def hit_rate_years(d):
    d = d.copy()
    d["year"] = d["year"].astype(str).str.replace(r"2023\+.*", "2023+ test", regex=True)
    order = list(d.year)
    bars = alt.Chart(d).mark_bar(cornerRadius=3, color=T.BLUE).encode(x=alt.X("year:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelOverlap=False, labelLimit=140)),
                                                                      y=alt.Y("hit_rate:Q", scale=alt.Scale(domain=[0, 0.8]), title="Share of films that are Hits", axis=alt.Axis(format="%")),
                                                                      tooltip=["year", alt.Tooltip("hit_rate:Q", format=".0%"), "n"])
    err = alt.Chart(d).mark_errorbar(color=T.TEXT).encode(x=alt.X("year:N", sort=order), y=alt.Y("ci_low:Q", title=""), y2="ci_high:Q")
    return _style(bars + err, 280)


def reliability(d):
    diag = alt.Chart(pd.DataFrame({"a": [0, 1], "b": [0, 1]})).mark_line(color=T.SLATE, strokeDash=[4, 4]).encode(x="a:Q", y="b:Q")
    pts = alt.Chart(d).mark_line(point=True).encode(x=alt.X("mean_predicted:Q", scale=alt.Scale(domain=[0, 1]), title="Average predicted chance", axis=alt.Axis(format="%")),
                                                    y=alt.Y("observed_hit_rate:Q", scale=alt.Scale(domain=[0, 1]), title="Share that really were Hits", axis=alt.Axis(format="%")),
                                                    color=alt.Color("model:N", legend=alt.Legend(title=None)), tooltip=["model", "n", alt.Tooltip("mean_predicted:Q", format=".0%"), alt.Tooltip("observed_hit_rate:Q", format=".0%")])
    return _style(diag + pts, 330)


def explain_chart(le, describe):
    """Bars that push towards Hit (green) or Flop (orange), labelled in plain words. `describe(feature, value)` returns (name, text)."""
    d = le.copy()
    d["label"] = [f"{n} ({t})" for n, t in (describe(f, v) for f, v in zip(d["feature"], d["value"]))]
    d["direction"] = np.where(d["shap"] > 0, "Towards Hit", "Towards Flop")
    order = d.assign(s=d["shap"].abs()).sort_values("s", ascending=False)["label"].tolist()
    c = alt.Chart(d).mark_bar(cornerRadius=3, height=18).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=280, labelOverlap=False, ticks=False, domain=False)),
        x=alt.X("shap:Q", title="Pushes towards Flop (left) or towards Hit (right)"),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["Towards Hit", "Towards Flop"], range=[T.HIT, T.FLOP]), legend=alt.Legend(title=None, symbolType="square")),
        tooltip=[alt.Tooltip("label:N", title="Input"), alt.Tooltip("shap:Q", title="Push", format="+.3f"), alt.Tooltip("direction:N")])
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=T.TEXT, strokeWidth=2).encode(x="x:Q")
    return _style(zero + c.properties(height=alt.Step(34)))
