"""Indian movie Hit/Flop research dashboard. Predictive model, pre-release information only. Run:  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hitflop import artifacts as A  # noqa: E402
from hitflop.dashboard import panels as PN  # noqa: E402

st.set_page_config(page_title="Indian Movie Hit/Flop Predictor", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")


@st.cache_resource(show_spinner="Loading models...")
def get_artifact():
    return A.load()


art = get_artifact()
PN.header(art)
tab_pred, tab_what, tab_cmp, tab_lb, tab_lab, tab_about = st.tabs(["Predict", "What-if", "Compare films", "Leaderboard", "Research lab", "About & limits"])
with tab_pred:
    PN.render_predict(art)
with tab_what:
    PN.render_whatif(art)
with tab_cmp:
    PN.render_compare(art)
with tab_lb:
    PN.render_leaderboard()
with tab_lab:
    PN.render_lab()
with tab_about:
    PN.render_about(art)
PN.footer()
