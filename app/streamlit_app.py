"""Indian movie Hit/Flop research app. Predictive model, pre-release information only. Run:  streamlit run app/streamlit_app.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hitflop import artifacts as A, config as C, explain as X, predict as P  # noqa: E402

st.set_page_config(page_title="Indian Movie Hit/Flop Predictor", page_icon="🎬", layout="wide")
LANG_NAMES = {"hi": "Hindi", "ta": "Tamil", "te": "Telugu", "ml": "Malayalam", "kn": "Kannada"}
GENRES = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery", "Romance", "Science Fiction", "Thriller", "War"]


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
    st.dataframe(df.round(fmt), use_container_width=True, hide_index=True)


# ------------------------------------------------------------------ header
REPO_URL = "https://github.com/tonymadhuri2088-cmd/movie-hit-flop"
st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1200px;}
.hero {background: linear-gradient(120deg, #312e81 0%, #4f46e5 55%, #0ea5e9 100%); color: #fff; border-radius: 18px;
       padding: 28px 32px; margin-bottom: 18px; box-shadow: 0 6px 24px rgba(79,70,229,.25);}
.hero h1 {margin: 0 0 6px 0; font-size: 2.1rem; color: #fff; padding: 0;}
.hero p {margin: 0; font-size: 1.02rem; opacity: .93; line-height: 1.5;}
.kpis {display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 18px;}
.kpi {background: #f3f4fb; border: 1px solid #e3e5f5; border-radius: 14px; padding: 14px 16px;}
.kpi b {display: block; font-size: 1.55rem; color: #312e81;}
.kpi span {font-size: .82rem; color: #5b6075;}
.verdict {border-radius: 14px; padding: 16px 20px; margin: 8px 0 14px 0; font-size: 1.05rem;}
.verdict.hit {background: #e8f7ee; border: 1px solid #9bd6b0; color: #14532d;}
.verdict.flop {background: #fdecec; border: 1px solid #f1a9a9; color: #7f1d1d;}
.verdict b {font-size: 1.25rem;}
.footer {margin-top: 36px; padding-top: 14px; border-top: 1px solid #e3e5f5; color: #6b7086; font-size: .85rem;}
button[data-baseweb="tab"] {font-weight: 600;}
@media (max-width: 800px) {.kpis {grid-template-columns: repeat(2, 1fr);} .hero h1 {font-size: 1.6rem;}}
</style>
<div class="hero">
  <h1>🎬 Indian Movie Hit / Flop Predictor</h1>
  <p>Will a film earn at least <b>2&times; its budget</b>? A research prototype that uses only information available <b>before release</b>:
  language, budget, runtime, genres, release timing and (optionally) Wikipedia attention. It shows how the model reasons and where it is weak.</p>
</div>
<div class="kpis">
  <div class="kpi"><b>854</b><span>labelled films (2015&ndash;2026)</span></div>
  <div class="kpi"><b>19</b><span>model families compared</span></div>
  <div class="kpi"><b>~0.76</b><span>test AUC (0.50 = chance)</span></div>
  <div class="kpi"><b>Honest</b><span>top models are statistically tied</span></div>
</div>
""", unsafe_allow_html=True)
with st.sidebar:
    st.markdown("### About")
    st.write("Predicts whether an Indian film will earn at least 2x its budget, using pre-release information only. "
             "Built with leak-free chronological validation and a test set scored once.")
    st.markdown(f"[Source code on GitHub]({REPO_URL})")
    st.markdown("**How to use**\n1. Open **Predict** and enter the film's details.\n2. Press **Predict**.\n3. Read the chance, the verdict and the explanation bars.")
    st.caption("A research prototype, not a financial tool. Do not use it to decide whether to invest in a film.")
tab_pred, tab_cmp, tab_dash, tab_about = st.tabs(["Predict", "Model comparison", "Research dashboard", "About & limits"])

# ------------------------------------------------------------------ predict
with tab_pred:
    art = get_artifact()
    c1, c2 = st.columns([1, 1.2], gap="large")
    with c1:
        st.subheader("Film details")
        mode_label = st.radio("Model", ["Basic: no promotion data", "Buzz-enhanced: add Wikipedia page views"], key="mode",
                              help="Buzz-enhanced adds Wikipedia page views from the 90 days before release. Its test score was higher but not statistically convincing.")
        mode = "basic" if mode_label.startswith("Basic") else "buzz"
        name = st.text_input("Film title (optional)", "", key="title")
        lang = st.selectbox("Language", list(LANG_NAMES), format_func=lambda k: LANG_NAMES[k], key="lang")
        rdate = st.date_input("Release date", value=pd.Timestamp("2027-01-15").date(), key="rdate")
        budget_m = st.number_input("Budget (US dollars, millions)", min_value=0.0, value=8.0, step=0.5, key="budget",
                                   help="The training data stores budgets in US dollars. Convert rupees yourself at the rate you trust.")
        runtime = st.number_input("Runtime (minutes)", min_value=0, max_value=400, value=140, key="runtime")
        genres = st.multiselect("Genres", GENRES, default=["Action"], key="genres")
        movie = {"name": name or "new film", "language": lang, "release_date": str(rdate), "budget": budget_m * 1e6, "runtime": runtime, "genres": genres}
        if mode == "buzz":
            st.markdown("**Wikipedia page views of the film's article, before release**")
            movie["pv_90_31"] = st.number_input("Days −90 to −31 (total views)", min_value=0, value=20000, key="pv1")
            movie["pv_30_8"] = st.number_input("Days −30 to −8 (total views)", min_value=0, value=30000, key="pv2")
            movie["pv_7_1"] = st.number_input("Days −7 to −1 (total views)", min_value=0, value=25000, key="pv3")
        go = st.button("Predict", type="primary", key="go")
    with c2:
        st.subheader("Prediction")
        if go:
            try:
                out = P.predict(art, movie, mode)
            except P.InputError as e:
                st.error(f"Please check the inputs: {e}")
            else:
                a = art[mode]
                p = out["probability"]
                st.metric("Model-estimated chance of a Hit", f"{p:.0%}")
                st.progress(min(max(p, 0.0), 1.0))
                _cls = "hit" if str(out["label"]).lower().startswith("hit") else "flop"
                _icon = "✅" if _cls == "hit" else "⚠️"
                st.markdown(f"<div class='verdict {_cls}'>{_icon} Verdict at the model's decision threshold ({a['threshold']:.2f}): <b>{out['label']}</b></div>", unsafe_allow_html=True)
                warn = []
                if lang == "hi":
                    warn.append("Hindi films are where this model is weakest (validation AUC about 0.56, barely above chance).")
                if not (1e6 <= movie["budget"] <= 6e7):
                    warn.append("This budget is outside the range where most training films sit, so treat the estimate with extra caution.")
                if pd.Timestamp(rdate) < pd.Timestamp.today():
                    warn.append("The release date is in the past; a prediction should be made before release.")
                for w_ in warn:
                    st.warning(w_)
                st.caption(f"On the frozen 2023+ test films this model scored AUC {a['test_auc']:.2f} (95% CI {a['test_auc_ci'][0]:.2f}–{a['test_auc_ci'][1]:.2f}). "
                           f"It predicted an average Hit chance of {a['test_mean_predicted']:.0%} where the observed Hit rate was {a['test_observed_hit_rate']:.0%}, "
                           "so probabilities tend to run high for recent films.")
                st.markdown("#### Why the model says this")
                f = out["features"]
                vals, Z, names = X.shap_values(a["pipeline"], a["shap_background"], f[a["columns"]])
                le = X.local_explanation(vals[0], Z[0], names, top=8)
                le["effect"] = np.where(le.shap > 0, "pushes towards Hit", "pushes towards Flop")
                st.bar_chart(le.set_index("feature")["shap"], horizontal=True)
                st.dataframe(le[["feature", "value", "shap", "effect"]].round(3), use_container_width=True, hide_index=True)
                st.caption("These bars show how the **model** used each input for this film. They are predictive associations, **not causes**: "
                           "a low budget raising the chance of a Hit reflects that ROI is a ratio and cheaper films clear 2× more easily, not that cutting your budget would make a film succeed. The `year` bar mostly reflects that the Hit rate fell over time.")
        else:
            st.info("Enter the film's details and press **Predict**.")

# ------------------------------------------------------------------ comparison
with tab_cmp:
    st.subheader("Final test results (2023 onwards, scored once)")
    st.caption("The test set was frozen before any modelling and scored a single time with pre-declared models. Nothing was selected on it.")
    show_table("final/test_results.csv")
    st.subheader("Is any model convincingly better?")
    show_table("final/test_comparisons.csv", "Paired bootstrap of the AUC difference; q is Benjamini-Hochberg adjusted. An interval that includes 0 means the gain could be luck.")
    st.subheader("By language (test)")
    show_table("final/test_by_language.csv")
    st.subheader("Development-period comparison of feature sets (out-of-fold)")
    df = read_csv("phase3/3a_feature_sets_by_model.csv")
    if df is not None:
        st.dataframe(df.pivot(index="set", columns="model", values="auc").round(3), use_container_width=True)
    st.subheader("Ensembles and tuning")
    show_table("phase45/5_ensembles.csv")
    show_table("phase45/4_tuned_vs_default.csv")

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
        st.dataframe(t.round(3), use_container_width=True, hide_index=True)
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
        st.dataframe(pd.DataFrame([{"time": r["time_utc"], "phase": r["phase"], "experiment": r["name"], "notes": r["notes"]} for r in rows]), use_container_width=True, hide_index=True)

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
