"""Indian movie Hit/Flop research app. Predictive model, pre-release information only. Run:  streamlit run app/streamlit_app.py"""
import html
import json
import sys
from pathlib import Path

import numpy as np
import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hitflop import artifacts as A, config as C, explain as X, predict as P, uncertainty as UC  # noqa: E402


st.set_page_config(page_title="Indian Movie Hit/Flop Predictor", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")
LANG_NAMES = {"hi": "Hindi", "ta": "Tamil", "te": "Telugu", "ml": "Malayalam", "kn": "Kannada"}
GENRES = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery", "Romance", "Science Fiction", "Thriller", "War"]
INK, HIT, FLOP, MUTED = "#14232b", "#0f7b62", "#c8412b", "#7b8a8f"


@st.cache_resource(show_spinner="Loading models...")
def get_artifact():
    return A.load()


def read_csv(rel):
    p = C.OUT_DIR / rel
    return pd.read_csv(p) if p.exists() else None


def show_table(rel, caption=None, fmt=3):
    df = read_csv(rel)
    if df is None:
        st.info(f"Result file `{rel}` is not available in this copy of the project.")
        return
    if caption:
        st.caption(caption)
    df = df.rename(columns={"Unnamed: 0": "item"})
    st.dataframe(df.round(fmt), width="stretch", hide_index=True)


EXAMPLES = {
    "Big-budget Telugu action": {"lang": "te", "budget": 25.0, "runtime": 165, "genres": ["Action", "Drama"]},
    "Mid-budget Tamil thriller": {"lang": "ta", "budget": 8.0, "runtime": 135, "genres": ["Thriller", "Crime"]},
    "Low-budget Malayalam drama": {"lang": "ml", "budget": 2.0, "runtime": 125, "genres": ["Drama"]},
    "Hindi family comedy": {"lang": "hi", "budget": 10.0, "runtime": 140, "genres": ["Comedy", "Family"]},
}


def apply_example():
    ex = EXAMPLES.get(st.session_state.get("example"))
    if ex:
        st.session_state["lang"] = ex["lang"]
        st.session_state["budget"] = ex["budget"]
        st.session_state["runtime"] = ex["runtime"]
        st.session_state["genres"] = ex["genres"]


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


def explain_chart(le):
    """Bars that push towards Hit (green) or Flop (red), labelled in plain words."""
    d = le.copy()
    parts = [describe(f, v) for f, v in zip(d["feature"], d["value"])]
    d["label"] = [f"{n} ({t})" for n, t in parts]
    d["direction"] = np.where(d["shap"] > 0, "Towards Hit", "Towards Flop")
    d["strength"] = d["shap"].abs()
    order = d.sort_values("strength", ascending=False)["label"].tolist()
    grid_col = alt.condition("datum.value === 0", alt.value(INK), alt.value("#e3e8e2"))
    grid_w = alt.condition("datum.value === 0", alt.value(2), alt.value(1))
    return alt.Chart(d).mark_bar(cornerRadius=3, height=18).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=260, labelFontSize=13, labelColor=INK, labelOverlap=False, ticks=False, domain=False)),
        x=alt.X("shap:Q", title="Pushes towards Flop (left) or towards Hit (right)",
                axis=alt.Axis(grid=True, gridColor=grid_col, gridWidth=grid_w, domain=False)),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["Towards Hit", "Towards Flop"], range=[HIT, FLOP]), legend=alt.Legend(orient="top", title=None, labelFontSize=13, symbolType="square")),
        tooltip=[alt.Tooltip("label:N", title="Input"), alt.Tooltip("shap:Q", title="Push", format="+.3f"), alt.Tooltip("direction:N", title="Direction")],
    ).properties(height=alt.Step(34)).configure_view(strokeWidth=0)


def auc_chart(df, name_col, auc_col, lo_col, hi_col, height=None):
    d = df[[name_col, auc_col, lo_col, hi_col]].copy()
    d.columns = ["model", "auc", "lo", "hi"]
    order = d.sort_values("auc", ascending=False)["model"].tolist()
    y = alt.Y("model:N", sort=order, title=None, axis=alt.Axis(labelLimit=280, labelFontSize=12, labelColor=INK, ticks=False, domain=False))
    xs = alt.Scale(domain=[0.5, 0.9])
    rule = alt.Chart(d).mark_rule(color="#98a5a8", strokeWidth=3).encode(y=y, x=alt.X("lo:Q", scale=xs, title="Test AUC (dot) with 95% interval; 0.5 = chance", axis=alt.Axis(tickCount=8, grid=True, gridColor="#e3e8e2", domain=False)), x2="hi:Q")
    dot = alt.Chart(d).mark_circle(size=110, color=INK).encode(y=y, x="auc:Q", tooltip=["model", alt.Tooltip("auc:Q", format=".3f"), alt.Tooltip("lo:Q", format=".3f"), alt.Tooltip("hi:Q", format=".3f")])
    chance = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(color=FLOP, strokeDash=[4, 4]).encode(x=alt.X("x:Q", scale=xs))
    return (chance + rule + dot).properties(height=height or max(140, 28 * len(d))).configure_view(strokeWidth=0)


def ticket_html(title, meta, p, thr, label, model_label):
    """The result, drawn as a cinema ticket. All user text is escaped."""
    cls = "hit" if str(label).lower().startswith("hit") else "flop"
    return (
        f"<div class='ticket {cls}'>"
        f"<div class='tk-top'><span>Box-office estimate</span><span>{html.escape(model_label)}</span></div>"
        f"<div class='tk-main'>"
        f"<div class='stamp'>{html.escape(str(label).upper())}</div>"
        f"<div class='tk-title'>{html.escape(title)}</div>"
        f"<div class='tk-meta'>{html.escape(meta)}</div>"
        f"<div class='tk-num'><span class='n'>{p * 100:.0f}</span><span class='pc'>%</span></div>"
        f"<div class='tk-lab'>chance of earning at least 2&times; its budget</div>"
        f"<div class='track'><div class='fill' style='width:{p * 100:.1f}%'></div>"
        f"<div class='thr' style='left:{thr * 100:.1f}%'><span>Hit line {thr * 100:.0f}%</span></div></div>"
        f"</div>"
        f"<div class='tk-perf'></div>"
        f"<div class='tk-foot'>A Hit is called when the chance reaches the Hit line. The line was set on past films, not to flatter this one.</div>"
        f"</div>")


# ------------------------------------------------------------------ look and feel
REPO_URL = "https://github.com/tonymadhuri2088-cmd/movie-hit-flop"
st.markdown("""
<style>
@font-face {font-family: "Archivo VF"; src: url("/app/static/archivo-wdth.woff2") format("woff2"); font-weight: 100 900; font-display: swap;}
@font-face {font-family: "Barlow Cond"; src: url("/app/static/barlow-condensed-800.woff2") format("woff2"); font-weight: 800; font-display: swap;}
@font-face {font-family: "Barlow Cond"; src: url("/app/static/barlow-condensed-700.woff2") format("woff2"); font-weight: 700; font-display: swap;}
:root {--paper:#f3f5f2; --panel:#ffffff; --line:#d9dfd8; --ink:#14232b; --muted:#5a6a70; --gold:#f2b632;
       --hit:#0f7b62; --flop:#c8412b; --hit-lit:#4fd1a5; --flop-lit:#ff7a59;
       --head:"Barlow Cond","Arial Narrow","Roboto Condensed",sans-serif;}
.stApp, .stApp p, .stApp label, .stApp li, .stApp td, .stApp th, .stApp button, .stApp input, .stApp textarea,
.stApp a {font-family: "Archivo VF", "Segoe UI", system-ui, sans-serif;}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {font-family: var(--head);}
.stApp [data-testid="stIconMaterial"], .stApp .material-symbols-rounded {font-family: "Material Symbols Rounded" !important;}
.block-container {padding-top: 3.4rem !important; padding-bottom: 2rem !important; max-width: 1180px;}
h2, h3 {font-weight: 800 !important; font-size: 1.75rem !important; line-height: 1.15 !important; letter-spacing: .005em; padding: .1rem 0 .3rem 0 !important; font-synthesis: none;}

.top {display:flex; align-items:flex-start; justify-content:space-between; gap:16px; flex-wrap:wrap;}
.logo {font-family:var(--head); font-weight:800; font-size:3rem; line-height:1.15; padding-top:.06em; color:var(--ink); letter-spacing:.005em; font-synthesis:none;}
.logo i {font-style:normal; color:var(--flop);}
.top a {color:var(--ink); font-weight:600; text-decoration:underline; text-underline-offset:3px; font-size:.92rem; margin-top:.9rem;}
.lede {font-size:1.02rem; color:var(--muted); max-width: 70ch; margin: 2px 0 12px 0; line-height:1.45;}
.facts {display:flex; flex-wrap:wrap; gap:6px 26px; border-top:2px solid var(--ink); border-bottom:1px solid var(--line); padding:9px 0; margin-bottom:6px; align-items:baseline;}
.facts div {font-size:.88rem; color:var(--muted); line-height:1.3;}
.facts b {font-family:var(--head); font-weight:800; font-size:1.35rem; color:var(--ink); margin-right:6px; font-synthesis:none;}

[data-testid="stVerticalBlockBorderWrapper"] {background: var(--panel); border-color: var(--line) !important; border-radius: 14px;}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {gap: .7rem;}
div[data-testid="stButton"] button {font-weight:700; padding:.6rem 1rem; border-radius:10px;}
button[data-baseweb="tab"] p {font-weight:650; font-size:.98rem;}

.ticket {position:relative; background:var(--ink); color:#f3f5f2; border-radius:16px; margin: 2px 0 10px 0; box-shadow: 0 10px 28px rgba(20,35,43,.2);}
.tk-top {display:flex; justify-content:space-between; gap:12px; background:var(--gold); color:var(--ink); font-weight:700; font-size:.82rem; padding:8px 22px; border-radius:16px 16px 0 0;}
.tk-main {position:relative; padding:16px 22px 14px 22px;}
.tk-title {font-family:var(--head); font-weight:800; font-size:1.9rem; line-height:1.1; padding-right:130px; overflow-wrap:anywhere; font-synthesis:none;}
.tk-meta {color:#b9c6c9; font-size:.9rem; margin-top:4px; padding-right:130px; line-height:1.35;}
.tk-num {margin-top:10px; line-height:.95; display:flex; align-items:baseline;}
.tk-num .n {font-family:var(--head); font-weight:800; font-size:5.4rem; line-height:.95; font-variant-numeric: tabular-nums; font-synthesis:none;}
.tk-num .pc {font-family:var(--head); font-weight:700; font-size:2.2rem; margin-left:4px; color:#b9c6c9;}
.tk-lab {color:#d6dfe1; font-size:.95rem; margin-top:2px;}
.track {position:relative; height:11px; background:rgba(255,255,255,.16); border-radius:8px; margin:30px 4px 4px 0;}
.fill {height:100%; border-radius:8px; background:#f3f5f2;}
.thr {position:absolute; top:-8px; bottom:-8px; width:3px; background:var(--gold); border-radius:2px; transform:translateX(-1px);}
.thr span {position:absolute; bottom:calc(100% + 3px); left:50%; transform:translateX(-50%); white-space:nowrap; font-size:.75rem; font-weight:700; color:var(--gold);}
.stamp {position:absolute; top:14px; right:20px; transform:rotate(-7deg); border:3px solid; border-radius:8px; padding:0 12px; font-family:var(--head); font-weight:800; font-size:2.1rem; line-height:1.25; letter-spacing:.04em; font-synthesis:none;}
.ticket.hit .stamp {color:var(--hit-lit); border-color:var(--hit-lit);}
.ticket.flop .stamp {color:var(--flop-lit); border-color:var(--flop-lit);}
.tk-perf {position:relative; border-top:2px dashed rgba(255,255,255,.28); margin:0 18px;}
.tk-perf::before, .tk-perf::after {content:""; position:absolute; top:-12px; width:22px; height:22px; border-radius:50%; background:var(--paper);}
.tk-perf::before {left:-29px;} .tk-perf::after {right:-29px;}
.tk-foot {padding:10px 22px 14px 22px; color:#aebcbf; font-size:.8rem; line-height:1.4;}

.ghost {border:2px dashed #aeb9b4; border-radius:16px; padding:22px 24px 18px 24px; color:var(--muted); background:rgba(255,255,255,.5);}
.ghost .g-title {font-family:var(--head); font-weight:800; font-size:1.8rem; line-height:1.15; color:var(--ink); margin-bottom:6px; font-synthesis:none;}
.ghost p {margin:0 0 8px 0; line-height:1.5; font-size:.98rem;}
.ghost ul {margin:4px 0 10px 18px; padding:0; line-height:1.55;}
.ghost .small {font-size:.85rem;}

.note {background:#fff7e0; border-left:4px solid var(--gold); border-radius:6px; padding:12px 16px; margin:8px 0 14px 0; line-height:1.5; color:var(--ink);}
.footer a {color:var(--ink); font-weight:600;}
.footer {margin-top:30px; padding-top:12px; border-top:1px solid var(--line); color:#5a6a70; font-size:.85rem;}
@media (max-width: 800px) {
  .logo {font-size:2.5rem;} .tk-num .n {font-size:4.6rem;} .tk-title {padding-right:0; padding-top:46px;} .tk-meta {padding-right:0;}
  .stamp {font-size:1.7rem; top:10px;} .facts {gap:4px 18px;}
}
@media (prefers-reduced-motion: reduce) {* {transition: none !important; animation: none !important;}}
</style>
""", unsafe_allow_html=True)
st.markdown(f"""
<div class="top"><div class="logo">Hit or <i>Flop</i></div><a href="{REPO_URL}" target="_blank">Source code on GitHub</a></div>
<p class="lede">Will an Indian film earn at least twice its budget? Enter what is known before release to see the model's estimate and the reasons behind it.</p>
<div class="facts">
  <div><b>854</b>labelled films, 2015 to 2026</div>
  <div><b>19</b>models compared fairly</div>
  <div><b>~0.76</b>test AUC (0.50 is guessing)</div>
  <div><b>No clear winner</b>top models are statistically tied</div>
</div>
""", unsafe_allow_html=True)
tab_pred, tab_cmp, tab_lb, tab_dash, tab_about = st.tabs(["Predict", "Model comparison", "19-model leaderboard", "Research dashboard", "About & limits"])

# ------------------------------------------------------------------ predict
with tab_pred:
    art = get_artifact()
    c1, c2 = st.columns([1, 1.15], gap="large")
    with c1:
        with st.container(border=True):
            st.subheader("Film details")
            st.selectbox("Quick example (invented films, not real titles)", ["Choose an example..."] + list(EXAMPLES), key="example", on_change=apply_example)
            name = st.text_input("Film title (optional)", "", key="title", placeholder="For example: My new film")
            lang = st.selectbox("Language", list(LANG_NAMES), format_func=lambda k: LANG_NAMES[k], key="lang")
            rdate = st.date_input("Release date", value=pd.Timestamp("2027-01-15").date(), key="rdate")
            b1, b2 = st.columns(2)
            with b1:
                budget_m = st.number_input("Budget (US$ millions)", min_value=0.0, value=8.0, step=0.5, key="budget",
                                           help="The training data stores budgets in US dollars. Convert rupees yourself at the rate you trust.")
            with b2:
                runtime = st.number_input("Runtime (minutes)", min_value=0, max_value=400, value=140, key="runtime")
            genres = st.multiselect("Genres", GENRES, default=["Action"], key="genres")
            mode_label = st.radio("Model", ["Basic: no promotion data", "Buzz-enhanced: add Wikipedia page views"], key="mode",
                                  help="Buzz-enhanced adds Wikipedia page views from the 90 days before release. Its test score was higher but not statistically convincing.")
            mode = "basic" if mode_label.startswith("Basic") else "buzz"
            movie = {"name": name or "new film", "language": lang, "release_date": str(rdate), "budget": budget_m * 1e6, "runtime": runtime, "genres": genres}
            if mode == "buzz":
                st.markdown("**Wikipedia page views of the film's article, before release**")
                movie["pv_90_31"] = st.number_input("Days −90 to −31 (total views)", min_value=0, value=20000, key="pv1")
                movie["pv_30_8"] = st.number_input("Days −30 to −8 (total views)", min_value=0, value=30000, key="pv2")
                movie["pv_7_1"] = st.number_input("Days −7 to −1 (total views)", min_value=0, value=25000, key="pv3")
            go = st.button("Predict", type="primary", key="go", width="stretch")
    with c2:
        if go:
            try:
                out = P.predict(art, movie, mode)
            except P.InputError as e:
                st.error(f"Please check the inputs: {e}")
            else:
                a = art[mode]
                p = out["probability"]
                meta = f"{LANG_NAMES[lang]}, ${budget_m:g}M budget, {runtime} min, {', '.join(genres) or 'no genre chosen'}"
                st.markdown(ticket_html(movie["name"], meta, p, a["threshold"], out["label"], "Basic model" if mode == "basic" else "Buzz-enhanced model"), unsafe_allow_html=True)
                warn = []
                if lang == "hi":
                    warn.append("Hindi films are where this model is weakest (validation AUC about 0.56, barely above chance).")
                if pd.Timestamp(rdate) < pd.Timestamp.today():
                    warn.append("The release date is in the past; a prediction should be made before release.")
                if pd.Timestamp(rdate) > pd.Timestamp.today() + pd.DateOffset(years=2):
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
                           f"It predicted an average Hit chance of {a['test_mean_predicted']:.0%} where the observed Hit rate was {a['test_observed_hit_rate']:.0%}, "
                           "so probabilities tend to run high for recent films.")
                st.subheader("Why the model says this")
                f = out["features"]
                vals, Z, names = X.shap_values(a["pipeline"], a["shap_background"], f[a["columns"]])
                le = X.local_explanation(vals[0], Z[0], names, top=8)
                le["effect"] = np.where(le.shap > 0, "pushes towards Hit", "pushes towards Flop")
                st.altair_chart(explain_chart(le), width="stretch")
                with st.expander("Show the numbers"):
                    _t = le[["feature", "value", "shap", "effect"]].round(3).rename(columns={"feature": "model input", "shap": "push"})
                    st.dataframe(_t, width="stretch", hide_index=True)
                st.caption("These bars show how the **model** used each input for this film. They are predictive associations, **not causes**: "
                           "a low budget raising the chance of a Hit reflects that ROI is a ratio and cheaper films clear 2× more easily, not that cutting your budget would make a film succeed. The release-year bar mostly reflects that the Hit rate fell over time.")
                _rep = pd.DataFrame([{"film": movie["name"], "language": LANG_NAMES[lang], "release_date": str(rdate), "budget_usd_millions": budget_m,
                                      "runtime_min": runtime, "genres": "; ".join(genres), "model": mode, "chance_of_hit": round(p, 4), "verdict": out["label"]}])
                st.download_button("Download this prediction (CSV)", _rep.to_csv(index=False).encode("utf-8"), file_name="hit_flop_prediction.csv", mime="text/csv")
        else:
            st.markdown("""
<div class="ghost">
  <div class="g-title">Your estimate appears here</div>
  <p>Pick a quick example or fill in the film's details, then press <b>Predict</b>. You get:</p>
  <ul><li>the chance of earning at least twice the budget</li><li>a Hit or Flop verdict</li><li>the inputs that pushed the answer each way</li></ul>
  <p class="small">A research prototype built on 854 past films. It is not a financial tool; do not use it to decide whether to invest in a film.</p>
</div>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ comparison
with tab_cmp:
    st.subheader("Final test results (2023 onwards, scored once)")
    st.caption("The test set was frozen before any modelling and scored a single time with pre-declared models. Nothing was selected on it.")
    _tr = read_csv("final/test_results.csv")
    if _tr is not None:
        st.altair_chart(auc_chart(_tr, "model", "auc", "auc_ci_low", "auc_ci_high", height=170), width="stretch")
        st.caption("The three real models\u2019 intervals overlap heavily, so they cannot be told apart on the 276 test films. The dummy baseline (always predict the development Hit rate) sits at chance, 0.50.")
    show_table("final/test_results.csv")
    st.subheader("Is any model convincingly better?")
    show_table("final/test_comparisons.csv", "Paired bootstrap of the AUC difference; q is Benjamini-Hochberg adjusted. An interval that includes 0 means the gain could be luck.")
    st.subheader("By language (test)")
    show_table("final/test_by_language.csv")
    st.subheader("Development-period comparison of feature sets (out-of-fold)")
    df = read_csv("phase3/3a_feature_sets_by_model.csv")
    if df is not None:
        st.dataframe(df.pivot(index="set", columns="model", values="auc").round(3), width="stretch")
    st.subheader("Ensembles and tuning")
    show_table("phase45/5_ensembles.csv")
    show_table("phase45/4_tuned_vs_default.csv")

# ------------------------------------------------------------------ leaderboard
with tab_lb:
    st.subheader("19 model families, one fair comparison")
    st.caption("Course notebook results. The table is ranked by validation AUC (chronological cross-validation on the 578 pre-2023 films); the chart is ordered by test AUC. "
               "Test AUC is on the 276 frozen 2023+ films, with a 95% bootstrap interval. A model is never chosen by its test score.")
    _lbp = ROOT / "docs" / "course_notebook_results" / "all_models_leaderboard.csv"
    if _lbp.exists():
        lb = pd.read_csv(_lbp)
        lb["label"] = lb["model"] + " (" + lb["feature group"].str.split(":").str[0] + ")"
        k1, k2, k3 = st.columns(3)
        vb = lb.sort_values("validation AUC", ascending=False).iloc[0]
        k1.metric("Models compared", len(lb))
        k2.metric("Test AUC of the validation-best model", f"{vb['test AUC']:.3f}", help=f"{vb['model']}, chosen on validation only; its 95% interval is {vb['ci_low']:.3f} to {vb['ci_high']:.3f}. We never pick a model by its test score.")
        k3.metric("Test AUC range, all 19", f"{lb['test AUC'].min():.2f} to {lb['test AUC'].max():.2f}")
        st.altair_chart(auc_chart(lb, "label", "test AUC", "ci_low", "ci_high"), width="stretch")
        st.markdown("<div class='note'><b>Reading this chart:</b> the top models all overlap, so none is convincingly better than a simple logistic regression. "
                    "The neural networks are lower on validation and no better on test, as expected with only 578 training films.</div>", unsafe_allow_html=True)
        show = lb.drop(columns=["label"]).rename(columns={"ci_low": "test CI low", "ci_high": "test CI high"})
        st.dataframe(show.round(3), width="stretch", hide_index=True)
        _img = ROOT / "docs" / "course_notebook_results" / "08c_all_models_leaderboard.png"
        if _img.exists():
            st.image(str(_img), caption="Leaderboard figure from the course notebook")
    else:
        st.info("The 19-model leaderboard file is not available in this copy of the project.")

# ------------------------------------------------------------------ dashboard
with tab_dash:
    st.subheader("Data quality and selection bias")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Share of mature films that have a usable label, by language")
        show_table("phase1/audit_labelled_share_language.csv")
    with c2:
        st.caption("Share by release year")
        show_table("phase1/audit_labelled_share_year.csv")
    st.caption("Only about 1 in 5 films has both budget and revenue. The labelled films can be predicted from public information (language, runtime, Wikipedia attention) far better than chance, so they are not a random sample.")
    show_table("phase1/audit_labelled_vs_unlabelled.csv")
    st.subheader("Robustness")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Validation AUC by language")
        show_table("phase3/3d_language_auc.csv")
    with c2:
        st.caption("Validation AUC by walk-forward fold (time)")
        show_table("phase3/3d_fold_auc.csv")
    st.caption("Changing the Hit threshold (ROI ≥ 1.5, 2.0, 2.5)")
    t = read_csv("phase3/3c_roi_thresholds.csv")
    if t is not None:
        st.dataframe(t.round(3), width="stretch", hide_index=True)
    st.caption("Identical-movie comparison of feature sets (only films that have both Wikipedia and YouTube data)")
    show_table("phase3/3b_common_cohort.csv")
    st.subheader("What the model relies on (SHAP, predictive not causal)")
    c1, c2 = st.columns(2)
    for col, mode_ in ((c1, "basic"), (c2, "buzz")):
        with col:
            st.caption(f"{mode_} model")
            img = C.OUT_DIR / "phase6" / f"{mode_}_shap_summary.png"
            if img.exists():
                st.image(str(img))
            show_table(f"phase6/{mode_}_group_importance.csv")
    st.subheader("Experiment log")
    lp = C.LOG_PATH
    if lp.exists():
        rows = [json.loads(l) for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
        st.dataframe(pd.DataFrame([{"time": r["time_utc"], "phase": r["phase"], "experiment": r["name"], "notes": r["notes"]} for r in rows]), width="stretch", hide_index=True)

# ------------------------------------------------------------------ about
with tab_about:
    st.subheader("What this is")
    st.markdown(
        "- **Question:** before release, can public information tell us whether an Indian film will earn at least twice its budget?\n"
        "- **Data:** 4,273 films (2015–2026) from the team's files, of which **854** have both budget and revenue and are old enough (≥ 120 days) for the box office to be final.\n"
        "- **Honest answer so far:** moderately, with big caveats. Test AUC is about 0.74–0.78; no model is convincingly better than a simple logistic regression.")
    st.subheader("Limits you should know")
    st.markdown(
        "- **Selection bias:** only films with public budget *and* revenue are labelled; they are not a random sample of Indian cinema.\n"
        "- **Drift:** the Hit rate fell from 47% (before 2023) to 30% (2023 onwards). Probabilities therefore tend to be too high for new films.\n"
        "- **Hindi:** validation AUC about 0.56, close to guessing.\n"
        "- **Money values** are in US dollars as stored in the files; whether revenue is gross or net is not documented.\n"
        "- **Predictive, not causal:** the explanations describe the model, not how to make a film succeed.\n"
        "- **Deployed models are refit on all 854 labelled films**, including the 2023+ films; the reported test scores come from the version fitted on pre-2023 films only.\n"
        "- **Not a financial tool.** Do not use it to decide whether to invest in a film.")
    st.subheader("Models")
    art = get_artifact()
    for m, label in (("basic", "Basic"), ("buzz", "Buzz-enhanced")):
        a = art[m]
        st.markdown(f"- **{label}:** {a['model']} on feature set {a['feature_set']}, trained on {a['trained_on']} films released {a['date_range'][0]} to {a['date_range'][1]}.")

st.markdown(f"<div class='footer'>Indian Movie Hit / Flop Predictor &middot; research prototype &middot; "
            f"<a href='{REPO_URL}' target='_blank'>source code</a> &middot; predictions describe patterns in past films, not what will happen to yours.</div>",
            unsafe_allow_html=True)
