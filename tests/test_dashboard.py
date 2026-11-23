import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from hitflop import artifacts as A
from hitflop import config as C
from hitflop import predict as P
from hitflop.dashboard import charts as CH
from hitflop.dashboard import logic as L

APP = str(C.ROOT / "app" / "streamlit_app.py")
MOVIE = {"name": "t", "language": "te", "release_date": "2027-01-15", "budget": 8e6, "runtime": 140, "genres": ["Action"]}


@pytest.fixture(scope="module")
def art():
    return A.load()


@pytest.mark.parametrize("mode,extra", [("basic", {}), ("buzz", {"pv_90_31": 20000, "pv_30_8": 30000, "pv_7_1": 25000})])
def test_what_if_equals_a_full_prediction(art, mode, extra):
    movie = {**MOVIE, **extra}
    w = L.what_if(art, P.predict(art, movie, mode)["features"], mode)
    for key, change, row in (("budget", {"budget": w["budget"].x.iloc[4] * 1e6}, w["budget"].iloc[4]), ("runtime", {"runtime": 120}, w["runtime"][w["runtime"].x == 120].iloc[0]),
                             ("language", {"language": "ml"}, w["language"][w["language"].x == "ml"].iloc[0]), ("month", {"release_date": "2027-07-15"}, w["month"].iloc[6])):
        assert abs(P.predict(art, {**movie, **change}, mode)["probability"] - row.probability) < 1e-9, key
    assert set(w) == {"budget", "runtime", "language", "month", "current"}
    assert all(((w[k].probability >= 0) & (w[k].probability <= 1)).all() for k in ("budget", "runtime", "language", "month"))


def test_what_if_marks_values_the_model_never_saw(art):
    f = P.predict(art, MOVIE, "basic")["features"]
    w = L.what_if(art, f, "basic")
    assert (~w["budget"].in_training_range).any() and w["budget"].in_training_range.any()


def test_compare_films_scores_each_film_and_reports_bad_input(art):
    films = [MOVIE, {**MOVIE, "name": "bad", "runtime": 5}, {**MOVIE, "name": "hindi", "language": "hi", "budget": 3e6}]
    res = L.compare_films(art, films, "basic")
    assert "error" in res[1] and "runtime" in res[1]["error"]
    ok = [r for r in res if "error" not in r]
    assert len(ok) == 2 and all(0 <= r["probability"] <= 1 and len(r["drivers"]) == 5 for r in ok)
    fr = L.comparison_frame(res)
    assert list(fr.film) == ["t", "hindi"] and set(fr.verdict) <= {"Hit", "Flop"}


def test_leaderboard_orders_by_the_chosen_metric():
    lb = pd.read_csv(C.ROOT / "docs" / "course_notebook_results" / "all_models_leaderboard.csv")
    for metric in ("validation AUC", "test AUC"):
        d = L.leaderboard(lb, metric)
        assert d[metric].is_monotonic_decreasing and list(d["rank"]) == list(range(1, len(d) + 1))
    assert 0 <= L.tied_share(L.leaderboard(lb)) <= 1


def test_charts_build():
    lb = pd.read_csv(C.ROOT / "docs" / "course_notebook_results" / "all_models_leaderboard.csv")
    CH.auc_forest(lb, "model", "test AUC", "ci_low", "ci_high").to_dict()
    CH.scatter_val_test(lb).to_dict()


def test_compare_tab_and_lab_render_without_error():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.button(key="cmp_go").click().run()
    assert not at.exception
    assert any("Hit line" in c for c in [x.value for x in at.caption]) or any("Gold tick" in x.value for x in at.caption)
    labels = [t.label for t in at.tabs]
    for lab in ("Claims ledger", "Feature groups", "Drift", "Calibration and warnings", "Languages", "Data quality", "Experiment log"):
        assert lab in labels


def test_whatif_follows_the_predict_inputs():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.selectbox(key="lang").select("ml").run()
    assert not at.exception
    assert any("What would change the estimate" in s.value for s in at.subheader)
