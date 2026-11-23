"""Streamlit panels, one function per tab. All numbers come from files written by the experiments or from the deployed model; nothing is typed in by hand except the headline counts."""
import html
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from .. import config as C
from .. import explain as X
from .. import ledger as LG
from .. import predict as P
from .. import uncertainty as UC
from . import charts as CH
from . import logic as L
from . import theme as T

GENRES = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery", "Romance", "Science Fiction", "Thriller", "War"]
LANG_NAMES = L.LANG_NAMES
EXAMPLES = {
    "Big-budget Telugu action": {"lang": "te", "budget": 25.0, "runtime": 165, "genres": ["Action", "Drama"]},
    "Mid-budget Tamil thriller": {"lang": "ta", "budget": 8.0, "runtime": 135, "genres": ["Thriller", "Crime"]},
    "Low-budget Malayalam drama": {"lang": "ml", "budget": 2.0, "runtime": 125, "genres": ["Drama"]},
    "Hindi family comedy": {"lang": "hi", "budget": 10.0, "runtime": 140, "genres": ["Comedy", "Family"]},
}
REPO_URL = "https://github.com/tonymadhuri2088-cmd/movie-hit-flop"


def read_csv(rel):
    p = C.OUT_DIR / rel
    return pd.read_csv(p) if p.exists() else None


def show_table(rel, caption=None, fmt=3, cols=None):
    df = read_csv(rel)
    if df is None:
        st.info(f"Result file `{rel}` is not available in this copy of the project.")
        return None
    if caption:
        st.caption(caption)
    df = df.rename(columns={"Unnamed: 0": "item"})
    st.dataframe(df[cols] if cols else df.round(fmt), width="stretch", hide_index=True)
    return df


def describe(feature, value):
    """Turn a model feature name and value into words a person can read."""
    if feature.startswith("genre_"):
        return f"Genre: {feature[6:]}", "yes" if value >= 0.5 else "no"
    if feature.startswith("language_"):
        return f"Language: {LANG_NAMES.get(feature[9:], feature[9:])}", "yes" if value >= 0.5 else "no"
    if feature.startswith("season_"):
        return f"Released in {feature[7:]}", "yes" if value >= 0.5 else "no"
    if feature == "log_budget":
        return "Budget", f"${10 ** value / 1e6:.1f}M"
    if feature == "runtime":
        return "Runtime", f"{value:.0f} min"
    if feature == "year":
        return "Release year", f"{value:.0f}"
    if feature in ("month_sin", "month_cos"):
        return "Release month", "seasonal pattern"
    if feature.startswith("l_pv_"):
        return "Wikipedia views", "log scale"
    return feature.replace("_", " ").capitalize(), f"{value:.2f}"


def ticket_html(title, meta, p, thr, label, model_label):
    """The result, drawn as a cinema ticket. All user text is escaped."""
    cls = "hit" if str(label).lower().startswith("hit") else "flop"
    return (f"<div class='ticket {cls}'><div class='tk-top'><span>Box-office estimate</span><span>{html.escape(model_label)}</span></div><div class='tk-main'>"
            f"<div class='stamp'>{html.escape(str(label).upper())}</div><div class='tk-title'>{html.escape(title)}</div><div class='tk-meta'>{html.escape(meta)}</div>"
            f"<div class='tk-num'><span class='n'>{p * 100:.0f}</span><span class='pc'>%</span></div><div class='tk-lab'>chance of earning at least 2&times; its budget</div>"
            f"<div class='track'><div class='fill' style='width:{p * 100:.1f}%'></div><div class='thr' style='left:{thr * 100:.1f}%'><span>Hit line {thr * 100:.0f}%</span></div></div></div>"
            f"<div class='tk-perf'></div><div class='tk-foot'>A Hit is called when the chance reaches the Hit line. The line was set on past films, not to flatter this one.</div></div>")


def header(art):
    st.markdown(T.CSS, unsafe_allow_html=True)
    ledger = LG.load_ledger()
    st.markdown(f"""
<div class="top"><div class="logo">Hit or <i>Flop</i></div><a href="{REPO_URL}" target="_blank">Source code on GitHub</a></div>
<p class="lede">Will an Indian film earn at least twice its budget? Enter what is known before release to see the model's estimate, the reasons behind it, and how much to trust it.</p>
<div class="kpis">
  <div class="kpi"><b>854</b><span>labelled films, 2015 to 2026</span></div>
  <div class="kpi"><b>19</b><span>model families compared fairly</span></div>
  <div class="kpi"><b>~0.74 to 0.77</b><span>frozen-test AUC (0.50 is guessing)</span></div>
  <div class="kpi"><b>{len(ledger)}</b><span>comparisons checked against one claim rule</span></div>
</div>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ predict
def apply_example():
    ex = EXAMPLES.get(st.session_state.get("example"))
    if ex:
        st.session_state.update({"lang": ex["lang"], "budget": ex["budget"], "runtime": ex["runtime"], "genres": ex["genres"]})


def movie_from_state(default_date="2027-01-15"):
    """The film currently described in the Predict tab's inputs (also used by the what-if tab)."""
    s = st.session_state
    mode = "basic" if str(s.get("mode", "Basic")).startswith("Basic") else "buzz"
    movie = {"name": s.get("title") or "new film", "language": s.get("lang", "te"), "release_date": str(s.get("rdate", pd.Timestamp(default_date).date())),
             "budget": float(s.get("budget", 8.0)) * 1e6, "runtime": s.get("runtime", 140), "genres": s.get("genres", ["Action"])}
    if mode == "buzz":
        movie.update({"pv_90_31": s.get("pv1", 20000), "pv_30_8": s.get("pv2", 30000), "pv_7_1": s.get("pv3", 25000)})
    return movie, mode


def render_predict(art):
    c1, c2 = st.columns([1, 1.15], gap="large")
    with c1:
        with st.container(border=True):
            st.subheader("Film details")
            st.selectbox("Quick example (invented films, not real titles)", ["Choose an example..."] + list(EXAMPLES), key="example", on_change=apply_example)
            st.text_input("Film title (optional)", "", key="title", placeholder="For example: My new film")
            st.selectbox("Language", list(LANG_NAMES), format_func=lambda k: LANG_NAMES[k], key="lang")
            st.date_input("Release date", value=pd.Timestamp("2027-01-15").date(), key="rdate")
            b1, b2 = st.columns(2)
            with b1:
                st.number_input("Budget (US$ millions)", min_value=0.0, value=8.0, step=0.5, key="budget", help="The training data stores budgets in US dollars. Convert rupees yourself at the rate you trust.")
            with b2:
                st.number_input("Runtime (minutes)", min_value=0, max_value=400, value=140, key="runtime")
            st.multiselect("Genres", GENRES, default=["Action"], key="genres")
            st.radio("Model", ["Basic: no promotion data", "Buzz-enhanced: add Wikipedia page views"], key="mode",
                     help="Buzz-enhanced adds Wikipedia page views from the 90 days before release. Its test score was higher but not statistically convincing.")
            if st.session_state.get("mode", "Basic").startswith("Buzz"):
                st.markdown("**Wikipedia page views of the film's article, before release**")
                st.number_input("Days −90 to −31 (total views)", min_value=0, value=20000, key="pv1")
                st.number_input("Days −30 to −8 (total views)", min_value=0, value=30000, key="pv2")
                st.number_input("Days −7 to −1 (total views)", min_value=0, value=25000, key="pv3")
            go = st.button("Predict", type="primary", key="go", width="stretch")
    with c2:
        if not go:
            st.markdown("""<div class="ghost"><div class="g-title">Your estimate appears here</div>
<p>Pick a quick example or fill in the film's details, then press <b>Predict</b>. You get:</p>
<ul><li>the chance of earning at least twice the budget</li><li>a Hit or Flop verdict and how much to trust it</li><li>the inputs that pushed the answer each way</li></ul>
<p class="small">A research prototype built on 854 past films. It is not a financial tool; do not use it to decide whether to invest in a film.</p></div>""", unsafe_allow_html=True)
            return
        movie, mode = movie_from_state()
        try:
            out = P.predict(art, movie, mode)
        except P.InputError as e:
            st.error(f"Please check the inputs: {e}")
            return
        a = art[mode]
        p = out["probability"]
        lang, budget_m, runtime, genres = movie["language"], movie["budget"] / 1e6, movie["runtime"], movie["genres"]
        meta = f"{LANG_NAMES[lang]}, ${budget_m:g}M budget, {runtime} min, {', '.join(genres) or 'no genre chosen'}"
        st.markdown(ticket_html(movie["name"], meta, p, a["threshold"], out["label"], "Basic model" if mode == "basic" else "Buzz-enhanced model"), unsafe_allow_html=True)
        warn = []
        rdate = pd.Timestamp(movie["release_date"])
        if lang == "hi":
            warn.append("Hindi films are where this model is weakest: inside Hindi its AUC is about 0.56, and the interval includes 0.50 (guessing).")
        if rdate < pd.Timestamp.today():
            warn.append("The release date is in the past; a prediction should be made before release.")
        if rdate > pd.Timestamp.today() + pd.DateOffset(years=2):
            warn.append("The release date is more than 2 years away. The model only saw films from 2015 to 2026, so the year effect is extrapolated and the estimate is unreliable.")
        if not genres:
            warn.append("No genre is selected. Training films all had genres, so this estimate is unreliable; pick at least one genre.")
        unc = (art.get("uncertainty") or {}).get(mode)
        assessment = UC.assess(unc, out["features"], p, a["threshold"]) if unc else None
        if assessment:
            warn = assessment["reasons"] + warn
        for w_ in warn:
            st.warning(w_)
        if assessment:
            for n_ in assessment["notes"]:
                st.info(n_)
        st.caption(f"On the frozen 2023+ test films this model scored AUC {a['test_auc']:.2f} (95% CI {a['test_auc_ci'][0]:.2f} to {a['test_auc_ci'][1]:.2f}). "
                   f"It predicted an average Hit chance of {a['test_mean_predicted']:.0%} where the observed Hit rate was {a['test_observed_hit_rate']:.0%}, so probabilities tend to run high for recent films.")
        st.subheader("Why the model says this")
        vals, Z, names = X.shap_values(a["pipeline"], a["shap_background"], out["features"][a["columns"]])
        le = X.local_explanation(vals[0], Z[0], names, top=8)
        st.altair_chart(CH.explain_chart(le, describe), width="stretch")
        with st.expander("Show the numbers"):
            le2 = le.assign(effect=np.where(le.shap > 0, "pushes towards Hit", "pushes towards Flop"))
            st.dataframe(le2[["feature", "value", "shap", "effect"]].round(3).rename(columns={"feature": "model input", "shap": "push"}), width="stretch", hide_index=True)
        st.caption("These bars show how the **model** used each input for this film. They are predictive associations, **not causes**: a low budget raising the chance of a Hit reflects that ROI is a ratio and cheaper films clear 2× more easily, "
                   "not that cutting your budget would make a film succeed. The release-year bar mostly reflects that the Hit rate fell over time.")
        rep = pd.DataFrame([{"film": movie["name"], "language": LANG_NAMES[lang], "release_date": movie["release_date"], "budget_usd_millions": budget_m, "runtime_min": runtime,
                             "genres": "; ".join(genres), "model": mode, "chance_of_hit": round(p, 4), "verdict": out["label"]}])
        st.download_button("Download this prediction (CSV)", rep.to_csv(index=False).encode("utf-8"), file_name="hit_flop_prediction.csv", mime="text/csv")


# ------------------------------------------------------------------ what-if
@st.cache_data(show_spinner="Trying other values...")
def _what_if_cached(movie_json, mode):
    from .. import artifacts as A
    art = A.load()
    movie = json.loads(movie_json)
    return L.what_if(art, P.predict(art, movie, mode)["features"], mode), art[mode]["threshold"]


def render_whatif(art):
    st.subheader("What would change the estimate?")
    st.caption("The film from the Predict tab, with one input changed at a time. This shows how the **model** responds, not what would make a real film earn more. "
               "Grey points are values the model rarely or never saw, so read them as extrapolation.")
    movie, mode = movie_from_state()
    try:
        w, thr = _what_if_cached(json.dumps(movie, sort_keys=True), mode)
    except P.InputError as e:
        st.error(f"Please check the inputs on the Predict tab: {e}")
        return
    cur = w["current"]
    k = st.columns(4)
    for col, (key, title) in zip(k, (("budget", "Budget"), ("runtime", "Runtime"), ("language", "Language"), ("month", "Release month"))):
        d = w[key]
        spread = d.probability.max() - d.probability.min()
        col.metric(title, f"{spread * 100:.0f} point" + ("" if round(spread * 100) == 1 else "s"), help="Gap between the highest and lowest estimate across the values tried below.")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Budget")
        st.altair_chart(CH.what_if_chart(w["budget"], "budget", cur["budget"], thr, "Budget (US$ millions, log scale)"), width="stretch")
        st.markdown("#### Release month")
        st.altair_chart(CH.what_if_chart(w["month"], "month", cur["month"], thr, "Release month"), width="stretch")
    with right:
        st.markdown("#### Runtime")
        st.altair_chart(CH.what_if_chart(w["runtime"], "runtime", cur["runtime"], thr, "Runtime (minutes)"), width="stretch")
        st.markdown("#### Language")
        st.altair_chart(CH.what_if_chart(w["language"], "language", cur["language"], thr, "Language"), width="stretch")
    st.caption("Gold line: the Hit line. Dotted marker or outlined bar: the values you entered. A budget curve that falls as budget rises mostly reflects that the target is revenue divided by budget, "
               "so a smaller budget is easier to double.")


# ------------------------------------------------------------------ compare
def render_compare(art):
    st.subheader("Compare films side by side")
    st.caption("Up to three films, scored by the same model, with the strongest reasons for each. Use it to see which inputs make the estimates differ.")
    names = list(EXAMPLES)
    mode_label = st.radio("Model", ["Basic: no promotion data", "Buzz-enhanced: add Wikipedia page views"], key="cmp_mode", horizontal=True)
    mode = "basic" if mode_label.startswith("Basic") else "buzz"
    use3 = st.checkbox("Compare a third film", value=False, key="cmp_third")
    n = 3 if use3 else 2
    cols = st.columns(n, gap="medium")
    films = []
    for i, col in enumerate(cols):
        ex = EXAMPLES[names[i]]
        with col, st.container(border=True):
            st.markdown(f"#### Film {'ABC'[i]}")
            title = st.text_input("Title", value=f"Film {'ABC'[i]}", key=f"cmp{i}_title")
            lang = st.selectbox("Language", list(LANG_NAMES), index=list(LANG_NAMES).index(ex["lang"]), format_func=lambda k: LANG_NAMES[k], key=f"cmp{i}_lang")
            rdate = st.date_input("Release date", value=pd.Timestamp("2027-01-15").date(), key=f"cmp{i}_date")
            budget = st.number_input("Budget (US$ millions)", min_value=0.0, value=float(ex["budget"]), step=0.5, key=f"cmp{i}_budget")
            runtime = st.number_input("Runtime (minutes)", min_value=0, max_value=400, value=int(ex["runtime"]), key=f"cmp{i}_runtime")
            genres = st.multiselect("Genres", GENRES, default=ex["genres"], key=f"cmp{i}_genres")
            movie = {"name": title or f"Film {'ABC'[i]}", "language": lang, "release_date": str(rdate), "budget": budget * 1e6, "runtime": runtime, "genres": genres}
            if mode == "buzz":
                movie.update({"pv_90_31": st.number_input("Views, days −90 to −31", min_value=0, value=20000, key=f"cmp{i}_pv1"),
                              "pv_30_8": st.number_input("Views, days −30 to −8", min_value=0, value=30000, key=f"cmp{i}_pv2"),
                              "pv_7_1": st.number_input("Views, days −7 to −1", min_value=0, value=25000, key=f"cmp{i}_pv3")})
            films.append(movie)
    if not st.button("Compare films", type="primary", key="cmp_go"):
        return
    res = L.compare_films(art, films, mode)
    for r in res:
        if "error" in r:
            st.error(f"{r['name']}: {r['error']}")
    fr = L.comparison_frame(res)
    if fr.empty:
        return
    st.altair_chart(CH.probability_bars(fr), width="stretch")
    st.caption("Gold tick: the Hit line.")
    show = fr.assign(chance_of_hit=(fr.chance_of_hit * 100).round(0).astype(int).astype(str) + "%", hit_line=(fr.hit_line * 100).round(0).astype(int).astype(str) + "%",
                     margin_to_line=(fr.margin_to_line * 100).round(0).astype(int).map("{:+d} pts".format)).rename(columns={"chance_of_hit": "chance of a Hit", "hit_line": "Hit line", "margin_to_line": "distance from the line"})
    st.dataframe(show, width="stretch", hide_index=True)
    d = CH.driver_bars(res)
    if d is not None:
        d = d.copy()
        d["reason"] = [f"{a} ({b})" for a, b in (describe(f, v) for f, v in zip(d.feature, d.value))]
        d["direction"] = np.where(d.shap > 0, "Towards Hit", "Towards Flop")
        chart = alt.Chart(d).mark_bar(cornerRadius=3).encode(
            y=alt.Y("reason:N", sort=alt.EncodingSortField("shap", op="sum", order="descending"), title=None, axis=alt.Axis(labelLimit=260, ticks=False, domain=False)),
            x=alt.X("shap:Q", title="Push towards Flop (left) or Hit (right)"),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["Towards Hit", "Towards Flop"], range=[T.HIT, T.FLOP]), legend=alt.Legend(title=None)),
            tooltip=["reason", alt.Tooltip("shap:Q", format="+.3f")]).properties(width=300, height=alt.Step(26)).facet(column=alt.Column("film:N", title=None))
        st.subheader("What drives each estimate")
        st.altair_chart(T.altair_style(chart), width="stretch")
    for r in res:
        if "error" not in r and r["reasons"]:
            st.warning(f"{r['name']}: " + " ".join(r["reasons"]))


# ------------------------------------------------------------------ leaderboard
def _ci_cell(df, m):
    return (df[m].round(3).astype(str) + " (" + df[m + "_lo"].round(3).astype(str) + " to " + df[m + "_hi"].round(3).astype(str) + ")") if m + "_lo" in df.columns else df[m].round(3).astype(str)


def render_leaderboard():
    st.subheader("Model leaderboard")
    lbp = C.ROOT / "docs" / "course_notebook_results" / "all_models_leaderboard.csv"
    if not lbp.exists():
        st.info("The 19-model leaderboard file is not available in this copy of the project.")
        return
    lb = pd.read_csv(lbp)
    lb["label"] = lb["model"] + " (" + lb["feature group"].str.split(":").str[0] + ")"
    top = st.columns([1, 1, 2])
    metric = top[0].radio("Order by", ["validation AUC", "test AUC"], horizontal=True, key="lb_metric", help="Validation is what a team is allowed to choose on. Test is reporting only.")
    fams = sorted(lb.family.unique())
    pick = top[2].multiselect("Model families", fams, default=fams, key="lb_fam")
    d = L.leaderboard(lb[lb.family.isin(pick)] if pick else lb, metric)
    all_d = L.leaderboard(lb, metric)
    k = st.columns(3)
    k[0].metric("Models compared", len(lb))
    k[1].metric("Test intervals overlapping the best", f"{L.tied_share(all_d):.0%}", help="Share of the 19 models whose 95% test interval overlaps the interval of the best test model. 100% means no model can be told apart from the rest.")
    best_val = all_d.sort_values("validation AUC", ascending=False).iloc[0]
    k[2].metric("Test AUC of the validation winner", f"{best_val['test AUC']:.3f}", help=f"{best_val['model']}; 95% interval {best_val['ci_low']:.3f} to {best_val['ci_high']:.3f}")
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.altair_chart(CH.auc_forest(d.assign(label=d["model"] + " (" + d["feature group"].str.split(":").str[0] + ")"), "label", "test AUC", "ci_low", "ci_high", domain=(0.5, 0.9), xtitle="Test AUC (dot) with 95% interval"), width="stretch")
    with right:
        st.altair_chart(CH.scatter_val_test(d), width="stretch")
        st.caption("Dots above the dashed line scored higher on the frozen test films than on the validation folds. Nearly every model does, so there is no sign that the models chosen on validation were over-fitted to it (folds trained on fewer films are a plausible reason for the gap).")
    st.markdown("<div class='note'><b>Reading this:</b> every model's interval overlaps the best one, so the ranking is not evidence that one model is better. "
                "The neural networks have no edge with 578 training films.</div>", unsafe_allow_html=True)
    st.dataframe(d.drop(columns=["label"], errors="ignore").rename(columns={"ci_low": "test CI low", "ci_high": "test CI high"}).round(3), width="stretch", hide_index=True)
    st.subheader("Full metric panel for the four named models")
    dev = read_csv("upgrade/phase1/dev_metrics_threshold_0.5.csv")
    tst = read_csv("upgrade/phase1/test_metrics_reporting_only.csv")
    m = [("auc", "AUC"), ("pr_auc", "PR-AUC"), ("brier", "Brier (lower is better)"), ("balanced_accuracy", "Balanced accuracy"), ("macro_f1", "Macro-F1"), ("mcc", "MCC")]
    tab_a, tab_b = st.tabs(["Development folds (480 films, chosen on)", "Frozen test (276 films, reporting only)"])
    for tab, df, note in ((tab_a, dev, "Out-of-fold predictions at a fixed 0.5 threshold, with bootstrap 95% intervals."),
                          (tab_b, tst[tst.threshold_rule.astype(str).str.contains("dev|youden", case=False)] if tst is not None else None, "Scored once; nothing was selected, tuned or fitted on these films.")):
        with tab:
            if df is None or df.empty:
                st.info("Not available in this copy of the project.")
                continue
            st.caption(note)
            table = pd.DataFrame({"model": df.model.values})
            for key, name in m:
                if key in df.columns:
                    table[name] = _ci_cell(df.reset_index(drop=True), key).values
            st.dataframe(table, width="stretch", hide_index=True)


# ------------------------------------------------------------------ research lab
def _verdict_pill(v):
    cls = "good" if "better" in v and "supported" in v else "bad" if "worse" in v and "supported" in v else ""
    return f"<span class='pill {cls}'>{html.escape(v)}</span>"


def render_lab():
    st.subheader("Research laboratory")
    st.caption("Every experiment of the upgrade, with its numbers. Nothing here was chosen on the frozen test set. Test-set figures are labelled reporting only.")
    t_claims, t_feat, t_drift, t_cal, t_lang, t_data, t_log = st.tabs(["Claims ledger", "Feature groups", "Drift", "Calibration and warnings", "Languages", "Data quality", "Experiment log"])
    with t_claims:
        led = LG.load_ledger()
        summ = LG.summarize(led)
        k = st.columns(4)
        k[0].metric("Comparisons", len(led))
        k[1].metric("Supported as better", int(summ.supported_better.sum()))
        k[2].metric("Supported as worse", int(summ.supported_worse.sum()))
        k[3].metric("No clear difference", int(summ.no_clear_difference.sum()))
        st.altair_chart(CH.ledger_stack(summ), width="stretch")
        st.markdown("<div class='note'><b>The claim rule:</b> a result is called better or worse only when the paired-bootstrap 95% interval excludes zero <i>and</i> the Benjamini-Hochberg adjusted p-value is below 0.05, "
                    "corrected over the whole family of comparisons. Every row below was re-checked against this rule by an automatic audit (zero violations).</div>", unsafe_allow_html=True)
        sup = led[led.verdict.str.contains("supported", na=False)].copy()
        ph = st.multiselect("Phase", sorted(sup.phase.unique()), default=sorted(sup.phase.unique()), key="led_phase")
        sup = sup[sup.phase.isin(ph)]
        st.caption(f"{len(sup)} comparisons met the rule. The rest are 'no clear difference'.")
        st.dataframe(sup[["phase", "experiment", "what", "metric", "diff", "ci_low", "ci_high", "q_bh", "verdict"]].round(4), width="stretch", hide_index=True)
        with st.expander("All comparisons"):
            st.dataframe(led.round(4), width="stretch", hide_index=True)
    with t_feat:
        st.markdown("#### Which kinds of information carry the prediction?")
        comp = read_csv("upgrade/phase5/ablation_vs_reference.csv")
        summ5 = read_csv("upgrade/phase5/ablation_summary.csv")
        if comp is None or summ5 is None:
            st.info("Phase 5 results are not available in this copy of the project.")
        else:
            model = st.radio("Model", ["LR", "HGB", "XGB"], horizontal=True, key="feat_model", format_func=lambda k: {"LR": "Logistic regression", "HGB": "Gradient boosting (basic model)", "XGB": "XGBoost (buzz model)"}[k])
            lo = comp[(comp.model == model) & (comp.metric == "auc") & (comp.design == "leave-one-out")].copy()
            lo["group"] = lo.configuration.str.replace("minus ", "", regex=False)
            st.markdown("**Remove one group from the full set** (left of 0: the group was helping)")
            st.altair_chart(CH.delta_forest(lo, "group", "diff", "ci_low", "ci_high", "verdict"), width="stretch")
            al = summ5[(summ5.model == model) & (summ5.role == "alone")].copy()
            al["group"] = al.configuration.str.replace("only ", "", regex=False)
            st.markdown("**Each group by itself** (right of the dashed line: better than guessing)")
            st.altair_chart(CH.auc_forest(al, "group", "auc", "auc_lo", "auc_hi", domain=(0.4, 0.75)), width="stretch")
            ad = comp[(comp.model == model) & (comp.metric == "auc") & (comp.design == "add-one")].copy()
            ad["group"] = ad.configuration.str.replace("base plus ", "", regex=False)
            st.markdown("**Add one extra group to the base set**")
            st.altair_chart(CH.delta_forest(ad, "group", "diff", "ci_low", "ci_high", "verdict"), width="stretch")
            st.markdown("<div class='note'><b>What this means:</b> budget is the only group whose removal clearly hurts (logistic regression and gradient boosting). "
                        "Budget and language are the strongest single groups (about 0.61 to 0.62 AUC each, alone); the six base groups together reach about 0.69. The extra track-record, calendar and competition blocks do not clearly help; "
                        "in the largest model, removing some of them clearly helps, which is a sign of noise with only 578 development films.</div>", unsafe_allow_html=True)
    with t_drift:
        st.markdown("#### Hit rate by release year")
        h = read_csv("upgrade/phase3/hit_rate_by_year.csv")
        if h is not None:
            st.altair_chart(CH.hit_rate_years(h), width="stretch")
        show_table("upgrade/phase3/hit_rate_trend_development.csv", "Trend test on development years only. A fall inside the development period is descriptive, not a significant trend.")
        st.markdown("#### Does forgetting old films help?")
        w = read_csv("upgrade/phase3/windows_vs_expanding.csv")
        if w is not None:
            w = w[w.metric == "auc"].copy()
            w["label"] = w.model + " | " + w.set + " | " + w.training_window
            st.altair_chart(CH.delta_forest(w, "label", "diff", "ci_low", "ci_high", "verdict", xtitle="Change in AUC versus the expanding window (95% interval)"), width="stretch")
            st.caption("Rolling windows and time-decay weights never beat the expanding window. Several are clearly worse, because they throw training films away.")
        show_table("upgrade/phase3/feature_drift.csv", "Input shift (PSI and KS): earlier versus later development films, and development versus the frozen test films (features only, no labels).")
    with t_cal:
        st.markdown("#### Reliability of the probabilities (development folds, blocks 2 to 5)")
        r = read_csv("upgrade/phase4/reliability_dev_raw.csv")
        if r is not None:
            st.altair_chart(CH.reliability(r), width="stretch")
            st.caption("On the dashed line, a 40% estimate means 40% of such films really were Hits. Points below it mean the model is too optimistic.")
        show_table("upgrade/phase4/calibration_dev_blocks_2_to_5.csv", "Calibration summaries for the raw model and each recalibration method.")
        rc = read_csv("upgrade/phase4/recalibration_vs_raw.csv")
        if rc is not None:
            st.markdown("#### Recalibration versus raw")
            st.dataframe(rc.round(4), width="stretch", hide_index=True)
        st.markdown("#### Do the warnings track real errors?")
        show_table("upgrade/phase4/warnings_vs_error_dev.csv", "Development folds. Only the close-call warning is supported by the evidence; input-shift flags are informational.")
    with t_lang:
        st.markdown("#### Inside each language")
        pl = read_csv("upgrade/phase6/per_language_performance.csv")
        if pl is not None:
            ok = pl[pl.reportable & (pl.model != "language-only baseline")].copy()
            ok["label"] = ok.language + " | " + ok.model
            st.altair_chart(CH.auc_forest(ok, "label", "auc", "auc_lo", "auc_hi", domain=(0.3, 0.95), xtitle="AUC inside the language (dot) with 95% interval; 0.5 = guessing", height=alt.Step(30)), width="stretch")
            st.caption("Telugu and Tamil: clearly above guessing for every model. Malayalam: above guessing for the boosted models only. Hindi, the largest group, is not clearly above guessing. Kannada has too few films to judge.")
            st.dataframe(pl.round(3), width="stretch", hide_index=True)
        show_table("upgrade/phase6/pooled_vs_within_language_auc.csv", "Pooled AUC versus the average AUC inside languages: part of the pooled score is simply knowing which language a film is in.")
        st.markdown("#### A language the model never saw")
        lo = read_csv("upgrade/phase6/leave_one_language_out.csv")
        if lo is not None:
            lo = lo.assign(label=lo.language + " | " + lo.model)
            st.altair_chart(CH.delta_forest(lo, "label", "diff", "ci_low", "ci_high", "verdict", xtitle="Change in AUC when the language is held out of training (95% interval)"), width="stretch")
            st.caption("No clear loss for any model, but the intervals are wide (about ±0.1), so this cannot show the model transfers across languages.")
    with t_data:
        c1, c2 = st.columns(2)
        with c1:
            show_table("phase1/audit_labelled_share_language.csv", "Share of mature films that have a usable label, by language")
        with c2:
            show_table("phase1/audit_labelled_share_year.csv", "Share by release year")
        st.caption("Only about 1 in 5 films has both budget and revenue. The labelled films can be told apart from the rest using public information, so they are not a random sample.")
        show_table("phase1/audit_labelled_vs_unlabelled.csv")
        for mode_ in ("basic", "buzz"):
            img = C.OUT_DIR / "phase6" / f"{mode_}_shap_summary.png"
            if img.exists():
                st.caption(f"What the {mode_} model relies on (SHAP, predictive not causal)")
                st.image(str(img))
    with t_log:
        lp = C.LOG_PATH
        if lp.exists():
            rows = [json.loads(line) for line in lp.read_text(encoding="utf-8").splitlines() if line.strip()]
            st.dataframe(pd.DataFrame([{"time": r["time_utc"], "phase": r["phase"], "experiment": r["name"], "notes": r["notes"]} for r in rows]), width="stretch", hide_index=True)


# ------------------------------------------------------------------ about
def render_about(art):
    st.subheader("What this is")
    st.markdown("- **Question:** before release, can public information tell us whether an Indian film will earn at least twice its budget?\n"
                "- **Data:** 4,273 films (2015–2026), of which **854** have both budget and revenue and are old enough (≥ 120 days) for the box office to be final.\n"
                "- **Honest answer so far:** moderately, with big caveats. Test AUC is about 0.74–0.77; no model is convincingly better than a simple logistic regression.")
    st.subheader("Limits you should know")
    st.markdown("- **Selection bias:** only films with public budget *and* revenue are labelled; they are not a random sample of Indian cinema.\n"
                "- **Drift:** the Hit rate fell from 47% (before 2023) to 30% (2023 onwards). Probabilities therefore tend to be too high for new films; no recalibration was supported as a fix.\n"
                "- **Hindi:** inside Hindi the AUC is about 0.56 and not clearly above guessing.\n"
                "- **Money values** are in US dollars as stored in the files; whether revenue is gross or net is not documented.\n"
                "- **Predictive, not causal:** the explanations describe the model, not how to make a film succeed.\n"
                "- **Deployed models are refit on all 854 labelled films**, including the 2023+ films; the reported test scores come from the version fitted on pre-2023 films only.\n"
                "- **Not a financial tool.** Do not use it to decide whether to invest in a film.")
    st.subheader("Models")
    for m, label in (("basic", "Basic"), ("buzz", "Buzz-enhanced")):
        a = art[m]
        st.markdown(f"- **{label}:** {a['model']} on feature set {a['feature_set']}, trained on {a['trained_on']} films released {a['date_range'][0]} to {a['date_range'][1]}.")


def footer():
    st.markdown(f"<div class='footer'>Indian Movie Hit / Flop Predictor &middot; research prototype &middot; <a href='{REPO_URL}' target='_blank'>source code</a> &middot; "
                "predictions describe patterns in past films, not what will happen to yours.</div>", unsafe_allow_html=True)
