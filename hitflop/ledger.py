"""Upgrade Phase 7: one ledger of every comparison made during the upgrade, and an automatic audit of the claim rule.

`load_ledger` reads every comparison table written by phases 1 to 6, puts them in one shape and keeps the original verdict.
`audit` re-applies the claim rule to each row: a "supported" verdict needs a 95% interval that excludes 0 on the right side and a Benjamini-Hochberg q below 0.05.
Nothing here recomputes a statistic; it only collects, counts and checks what the experiment runners already wrote.
"""
import numpy as np
import pandas as pd

from . import config as C

UP = C.OUT_DIR / "upgrade"
SOURCES = [
    ("1 Evaluation", "Models versus logistic regression, development folds", "phase1/dev_comparisons_vs_logistic_regression.csv"),
    ("2 Minority class", "Imbalance methods versus the plain model", "phase2/natural_variant_vs_plain.csv"),
    ("2 Minority class", "Imbalance-method models versus logistic regression", "phase2/natural_vs_logistic_regression.csv"),
    ("2 Minority class", "Threshold rules versus Youden", "phase2/threshold_rules_vs_youden.csv"),
    ("2 Minority class", "Simulated 30% minority, per seed", "phase2/simulated_minority_per_seed_tests.csv"),
    ("3 Drift", "Training windows versus expanding window", "phase3/windows_vs_expanding.csv"),
    ("3 Drift", "Windowed models versus logistic regression", "phase3/windows_vs_logistic_regression.csv"),
    ("4 Calibration", "Recalibration versus raw probabilities", "phase4/recalibration_vs_raw.csv"),
    ("4 Calibration", "Warning flags versus prediction quality", "phase4/warnings_vs_error_dev.csv"),
    ("5 Feature ablation", "Feature groups removed or added", "phase5/ablation_vs_reference.csv"),
    ("6 Language", "Never-seen language versus the normal model", "phase6/leave_one_language_out.csv"),
]
DESCRIBE = ["model", "set", "variant", "spec", "rule", "training_window", "method", "configuration", "flag", "language", "seed"]


def _what(row):
    return " | ".join(str(row[c]) for c in DESCRIBE if c in row.index and pd.notna(row[c]))


def load_ledger():
    frames = []
    for phase, experiment, rel in SOURCES:
        d = pd.read_csv(UP / rel)
        out = pd.DataFrame({"phase": phase, "experiment": experiment, "what": d.apply(_what, axis=1),
                            "metric": d["metric"] if "metric" in d.columns else "auc",
                            "diff": d["diff"], "ci_low": d["ci_low"], "ci_high": d["ci_high"], "q_bh": d["q_bh"] if "q_bh" in d.columns else np.nan, "verdict": d["verdict"]})
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def direction(verdict):
    v = str(verdict)
    if "supported" not in v:
        return 0
    return 1 if ("better" in v) else -1


def audit(ledger):
    """Rows whose wording is not licensed by their own numbers. Empty means every claim in the repository follows the rule."""
    bad = []
    for i, r in ledger.iterrows():
        d = direction(r.verdict)
        if d == 0:
            if r.verdict == "no clear difference" and pd.notna(r.q_bh) and r.q_bh < 0.05 and (r.ci_low > 0 or r.ci_high < 0):
                bad.append((i, "evidence of a difference but worded as none"))
            continue
        if not (r.q_bh < 0.05):
            bad.append((i, "q not below 0.05"))
        elif not (r.ci_low > 0 or r.ci_high < 0):
            bad.append((i, "interval includes 0"))
        elif "flagged films" not in r.verdict and ((d > 0 and not r.ci_low > 0) or (d < 0 and not r.ci_high < 0)):
            bad.append((i, "interval on the wrong side"))
    return pd.DataFrame(bad, columns=["row", "problem"])


def summarize(ledger):
    g = ledger.assign(better=ledger.verdict.str.contains("better", na=False) & ledger.verdict.str.contains("supported", na=False),
                      worse=ledger.verdict.str.contains("worse|do worse", na=False) & ledger.verdict.str.contains("supported", na=False))
    s = g.groupby(["phase", "experiment"], sort=False).agg(comparisons=("verdict", "size"), supported_better=("better", "sum"), supported_worse=("worse", "sum")).reset_index()
    s["no_clear_difference"] = s.comparisons - s.supported_better - s.supported_worse
    return s
