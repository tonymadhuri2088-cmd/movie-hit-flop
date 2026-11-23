import re

import pytest
from streamlit.testing.v1 import AppTest

from pathlib import Path
APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def _run():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def _ticket(at):
    """The result ticket is HTML; return its text, or None if no prediction is shown."""
    for m in at.markdown:
        if "class='ticket" in m.value:
            return m.value
    return None


def test_app_loads_without_exception():
    at = _run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Predict", "Model comparison", "19-model leaderboard", "Research dashboard", "About & limits"]


def test_basic_prediction_end_to_end():
    at = _run()
    at.button(key="go").click().run()
    assert not at.exception
    t = _ticket(at)
    assert t is not None
    p = float(re.search(r"<span class='n'>(\d+)</span>", t).group(1)) / 100
    assert 0 <= p <= 1
    assert "FLOP" in t or "HIT" in t


def test_buzz_prediction_end_to_end():
    at = _run()
    at.radio(key="mode").set_value("Buzz-enhanced: add Wikipedia page views").run()
    at.button(key="go").click().run()
    assert not at.exception
    assert _ticket(at) is not None


def test_invalid_budget_shows_readable_error_not_a_crash():
    at = _run()
    at.number_input(key="budget").set_value(0.01).run()
    at.button(key="go").click().run()
    assert not at.exception
    assert any("budget" in e.value for e in at.error)


def test_hindi_gets_a_weakness_warning():
    at = _run()
    at.selectbox(key="lang").set_value("hi").run()
    at.button(key="go").click().run()
    assert any("Hindi" in w.value for w in at.warning)


def test_no_genre_gets_a_warning():
    at = _run()
    at.multiselect(key="genres").set_value([]).run()
    at.button(key="go").click().run()
    assert not at.exception
    assert any("No genre" in w.value for w in at.warning)


def test_far_future_date_gets_a_warning():
    import datetime
    at = _run()
    at.date_input(key="rdate").set_value(datetime.date(2034, 1, 15)).run()
    at.button(key="go").click().run()
    assert not at.exception
    assert any("more than 2 years" in w.value for w in at.warning)


def test_quick_example_fills_the_form_and_predicts():
    at = _run()
    at.selectbox(key="example").set_value("Low-budget Malayalam drama").run()
    assert at.selectbox(key="lang").value == "ml"
    assert at.number_input(key="budget").value == 2.0
    at.button(key="go").click().run()
    assert not at.exception
    assert _ticket(at) is not None


def test_leaderboard_tab_shows_19_models():
    at = _run()
    assert not at.exception
    assert any(m.label == "Models compared" and m.value == "19" for m in at.metric)


def test_extreme_budget_shows_an_extrapolation_warning():
    at = _run()
    at.number_input(key="budget").set_value(900.0).run()
    at.button(key="go").click().run()
    assert not at.exception
    assert any("outside every budget" in w.value for w in at.warning)


def test_unusual_inputs_show_a_note_not_a_warning():
    at = _run()
    at.number_input(key="budget").set_value(0.15).run()      # inside the seen range, below the 1st percentile
    at.button(key="go").click().run()
    assert not at.exception
    assert any("extreme end" in i.value for i in at.info)
    assert not any("extreme end" in w.value for w in at.warning)
