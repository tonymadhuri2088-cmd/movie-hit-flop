# Indian Movie Hit / Flop: research-grade upgrade

Predicts whether an Indian film earns at least 2x its budget (**Hit**, ROI >= 2.0) using only information known before release.
This project extends the original notebook (`movie_hit_flop_neha_data.ipynb`, left unchanged). It reproduces the original cohort exactly
(854 labelled films, 578 development before 2023-01-01, 276 frozen test) and then adds stricter validation, new features, tuning, ensembles, calibration, explanations and an app.

## Headline results (honest version)
| Question | Answer |
|---|---|
| Final test AUC (2023+, scored once) | basic HGB 0.735; Wikipedia-buzz XGBoost 0.776; plain logistic regression 0.759; no-skill 0.50 |
| Is any model convincingly better than logistic regression? | **No.** All paired-bootstrap intervals include 0. |
| Do the new features help (Wikipedia per-day rates, YouTube ratios, festivals, competition, cast/crew history)? | **No.** 0 of 48 development comparisons convincing; adding everything made XGBoost worse. |
| Does Optuna tuning help? | **No.** It never beat the default settings (0 of 10). |
| Do ensembles help? | **No.** None convincingly beat the best single model; stacking was the worst. |
| Calibration | Fitting calibrators on earlier folds made probabilities worse, because the Hit rate drifts (47% before 2023, 30% after). Not used. Predictions run high: 0.34-0.40 predicted vs 0.30 observed. |
| Weak spot | Hindi (237 dev films) validation AUC 0.56; Telugu 0.73, Tamil 0.80, Malayalam 0.69. |
| Selection bias | Only 854 of 4,273 films have budget and revenue; they are predictable from public covariates (selection-model AUC 0.86), so they are not a random sample. |

## Layout
- `hitflop/` package: `data` (cohort, split, folds), `history` (leak-free cast/crew/studio records), `features`, `models`, `validation`, `metrics`, `calibration`, `tuning` (nested Optuna), `ensembles`, `explain` (SHAP), `predict`, `artifacts`, `experiment_log`, and phase runners.
- `tests/` 59 tests: cohort reproduction, leakage (perturb future labels and check earlier features do not move), validation folds, metrics, calibration, ensembles, tuning, featurizer-versus-pipeline equality, app end-to-end.
- `experiments/experiment_log.jsonl` append-only record of every experiment (settings, results, notes), including the pre-registration of the final models before test scoring.
- `notebooks/02_upgrade_experiments.ipynb` results and a worked nested-Optuna example.
- `app/streamlit_app.py` the app (Predict, Model comparison, Research dashboard, About & limits).
- `outputs/` all result tables and figures; `artifacts/` deployed models.

## Run
```
pip install -r requirements.txt
python -m pytest -q tests
python -m hitflop.pipeline_phase1; python -m hitflop.phase3; python -m hitflop.phase45   # re-creates the dev-only results
python -m hitflop.final            # scores the frozen test set ONCE (a lock file refuses a second run)
python -m hitflop.artifacts; python -m hitflop.phase6
streamlit run app/streamlit_app.py
```
Seeds are fixed (42). The Hit definition (ROI >= 2.0) is unchanged; 1.5 and 2.5 are sensitivity checks only (`outputs/phase3/3c_roi_thresholds.csv`).

## Method notes
- Validation: expanding-window walk-forward (5 outer folds on development data), nested inner chronological split for tuning, one pooled set of 480 out-of-fold predictions.
- Comparisons: paired bootstrap on identical films, Benjamini-Hochberg across the comparison family; "convincing" = interval above 0 and q < 0.05.
- Track records only use films released at least 120 days earlier; proven against a brute-force reference and by label-flip perturbation tests.
- Ensembles and calibrators for fold k are fitted only on out-of-fold predictions from folds before k.
- The deployed app models are refit on all 854 labelled films; the reported test scores are from the pre-2023 fit.

## Limits
Predictive, not causal (SHAP describes the model). Money is in US dollars as stored; gross vs net is undocumented. YouTube collection dates are missing, so YouTube features cannot be verified as pre-release. Holiday calendar: Pongal/Sankranti fixed at 14 January; regional festivals such as Onam are not included.
