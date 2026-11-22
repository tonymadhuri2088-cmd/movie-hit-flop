# Architecture and design decisions

## Pipeline
```mermaid
flowchart TD
    subgraph Data
      M[movies_full.csv<br/>4,273 films] --> L[Label: ROI = revenue / budget<br/>854 films have both]
      W[Wikipedia page views<br/>and infoboxes] --> F
      Y[yt_features.csv<br/>trailer statistics] --> F
    end
    L --> S{Chronological split}
    S -->|before 2023| DEV[578 development films]
    S -->|2023 onward| TEST[276 frozen test films]
    DEV --> F[Feature engineering<br/>leak-free history, calendar, buzz]
    F --> CV[Walk-forward cross-validation]
    CV --> CMP[19 models + nested Optuna tuning<br/>paired bootstrap + BH]
    CMP --> FINAL[Declared models fitted on development films]
    FINAL --> SCORE[Score frozen test ONCE<br/>TEST_SCORED.lock]
    TEST --> SCORE
    FINAL --> ART[Deployed artifacts<br/>refit on all 854 films]
    ART --> APP[Streamlit app<br/>prediction + SHAP]
```

## Key decisions
| Decision | Why |
|---|---|
| Chronological splits, never random | Random shuffling lets later films inform earlier ones and inflates scores. |
| Test set frozen and scored once | Prevents choosing a model by its test score. A lock file enforces it. |
| Track records use only films released at least 120 days earlier | Box-office results are not known sooner. Label-flip tests prove no leakage. |
| Paired bootstrap + Benjamini-Hochberg | Many comparisons were made, so adjusted p-values stop chance wins being reported as findings. |
| Hit threshold fixed at ROI 2.0 | 1.5 and 2.5 are reported only as sensitivity checks. |
| No probability calibration in the app | Calibrators fitted on earlier folds made probabilities worse because the Hit rate drifts. |
| Two model modes (basic, buzz-enhanced) | Wikipedia views only exist for some films; the app lets you choose and warns when a result is weak. |

## Code layout
- `hitflop/data.py` loads and cleans, `features.py` builds features, `history.py` computes leak-free track records.
- `validation.py` makes walk-forward folds, `metrics.py` computes AUC, PR-AUC, Brier and bootstrap intervals.
- `tuning.py` and `ensembles.py` hold nested Optuna tuning and stacking.
- `explain.py` wraps SHAP, `predict.py` serves the app, `final.py` scores the frozen test set once.
