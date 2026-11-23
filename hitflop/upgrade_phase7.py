"""Upgrade Phase 7 runner: compare old and new models without claiming insignificant gains. Reads the tables written by phases 1 to 6; computes nothing new.

Outputs (outputs/upgrade/phase7/): claims_ledger.csv, ledger_summary.csv, supported_findings.csv, best_looking_upgrades.csv; figure docs/figures/phase7_claims_ledger.png.
Run:  python -m hitflop.upgrade_phase7
"""

from . import config as C
from . import ledger as L
from . import protocol as P

OUT = C.OUT_DIR / "upgrade" / "phase7"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    led = L.load_ledger()
    bad = L.audit(led)
    assert bad.empty, f"claims not licensed by their own numbers:\n{led.loc[bad.row].assign(problem=bad.problem.values)}"
    led.to_csv(OUT / "claims_ledger.csv", index=False)
    summ = L.summarize(led)
    summ.to_csv(OUT / "ledger_summary.csv", index=False)
    sup = led[led.verdict.str.contains("supported", na=False)].sort_values(["phase", "diff"])
    sup.to_csv(OUT / "supported_findings.csv", index=False)
    # the largest AUC-style gain each upgrade idea ever showed, and what the claim rule says about it
    auc = led[led.metric.astype(str).str.startswith("auc") & led.phase.isin(["2 Minority class", "3 Drift", "4 Calibration", "5 Feature ablation"])]
    best = auc.sort_values("diff", ascending=False).groupby("phase").head(3)
    best.to_csv(OUT / "best_looking_upgrades.csv", index=False)
    figure(summ)
    P.log_run("phase7-claims-ledger", "upgrade-p7", {"source_tables": [s[2] for s in L.SOURCES], "rule": "interval excludes 0 and BH q < 0.05"},
              {"comparisons": int(len(led)), "supported_better": int(summ.supported_better.sum()), "supported_worse": int(summ.supported_worse.sum()),
               "no_clear_difference": int(summ.no_clear_difference.sum()), "audit_violations": 0},
              notes="Collects and audits existing results; no new statistic.", uses_test_set=False)
    print(summ.to_string(index=False))
    print("audit violations: 0")


def figure(summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = summ.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(s) + 1.6))
    y = range(len(s))
    ax.barh(y, s.supported_better, color="#2E7D5B", label="better (supported)")
    ax.barh(y, s.no_clear_difference, left=s.supported_better, color="#C9CED6", label="no clear difference")
    ax.barh(y, s.supported_worse, left=s.supported_better + s.no_clear_difference, color="#B8512A", label="worse (supported)")
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{r.phase}: {r.experiment}" for r in s.itertuples()], fontsize=8)
    ax.set_xlabel("Number of comparisons")
    ax.set_title("Every comparison in the upgrade, by what the evidence supports", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(C.ROOT / "docs" / "figures" / "phase7_claims_ledger.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
