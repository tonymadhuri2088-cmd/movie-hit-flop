import pandas as pd

from hitflop import ledger as L


def test_every_claim_in_the_repository_follows_the_rule():
    led = L.load_ledger()
    assert len(led) > 400
    assert L.audit(led).empty


def test_audit_catches_an_unlicensed_claim():
    led = pd.DataFrame([
        {"phase": "x", "experiment": "e", "what": "a", "metric": "auc", "diff": 0.02, "ci_low": -0.01, "ci_high": 0.05, "q_bh": 0.01, "verdict": "better (supported)"},      # interval includes 0
        {"phase": "x", "experiment": "e", "what": "b", "metric": "auc", "diff": 0.02, "ci_low": 0.01, "ci_high": 0.05, "q_bh": 0.20, "verdict": "better (supported)"},       # q too large
        {"phase": "x", "experiment": "e", "what": "c", "metric": "auc", "diff": 0.04, "ci_low": 0.02, "ci_high": 0.06, "q_bh": 0.01, "verdict": "no clear difference"},       # evidence hidden
        {"phase": "x", "experiment": "e", "what": "d", "metric": "auc", "diff": 0.04, "ci_low": 0.02, "ci_high": 0.06, "q_bh": 0.01, "verdict": "better (supported)"},       # fine
    ])
    assert list(L.audit(led).row) == [0, 1, 2]


def test_summary_counts_add_up():
    s = L.summarize(L.load_ledger())
    assert (s.comparisons == s.supported_better + s.supported_worse + s.no_clear_difference).all()
