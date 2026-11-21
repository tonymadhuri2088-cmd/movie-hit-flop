# What is done

A plain status page: what was built, what it showed, and what was not done.

## Done
| Area | What was done | Evidence |
|---|---|---|
| Problem and label | Hit = revenue / budget >= 2.0, using only information known before release | `hitflop/config.py`, README |
| Cohort and split | 854 labelled films; 578 development (before 2023-01-01) and 276 frozen test; seed 42 | `tests/test_cohort_and_splits.py` |
| Data audit | Missing values, duplicates, filter log, currency check, selection-bias analysis (only 854 of 4,273 films have budget and revenue) | `outputs/phase1/`, `docs/course_notebook_results/` |
| Features | Wikipedia page-view rates, YouTube ratios, festivals and holidays, release competition, cast/crew/studio track records (only earlier films, at least 120 days before release) | `hitflop/features.py`, `hitflop/history.py` |
| Leakage control | Chronological splits only; tests flip future labels and check earlier features do not change; shuffled-label check | `tests/test_history_leakage.py`, `09_shuffled_labels.png` |
| Models | 19 model families: 3 main classic, 13 extra (XGBoost, LightGBM, CatBoost, AdaBoost, Extra Trees, kNN, SVM, Naive Bayes, LDA, elastic-net, average ensemble, stacking, sklearn Gradient Boosting), 3 neural networks (MLP, GRU, 1D-CNN) | `docs/MODEL_LEADERBOARD.md` |
| Validation | Expanding-window chronological cross-validation, nested inner split for tuning | `hitflop/validation.py` |
| Statistics | Paired bootstrap on identical films and Benjamini-Hochberg correction; "convincing" = interval above 0 and q < 0.05 | `outputs/` tables |
| Tuning and ensembles | Nested Optuna tuning, stacking, average ensemble | `hitflop/tuning.py`, `hitflop/ensembles.py` |
| Calibration | Tested; fitted calibrators made things worse because the Hit rate drifts, so they are not used | `docs/course_notebook_results/12_calibration.png` |
| Explanation | SHAP and permutation importance | `13_shap_summary.png`, `hitflop/explain.py` |
| Frozen test | Scored once; lock file refuses a second scoring | `outputs/final/TEST_SCORED.lock` |
| Robustness | ROI thresholds 1.5 and 2.5, missing-buzz subset, language and year shifts, complete-case analysis, learning curve, YouTube timing check | `outputs/`, notebook |
| App | Streamlit app with Predict, Model comparison, Research dashboard and About & limits | `app/streamlit_app.py` |
| Tests and CI | 59 tests pass locally; GitHub Actions runs them on every push | `tests/`, `.github/workflows/ci.yml` |
| Course notebook | One notebook, all 19 models, run end to end in Colab: "All checks passed", total runtime 1881 s | `notebooks/01_course_notebook_19_models.ipynb` |

## What it showed
1. Hit/Flop can be predicted **moderately** before release: test AUC about 0.76 (95% CI roughly 0.70 to 0.83) versus 0.50 for no skill.
2. **No model is convincingly better than logistic regression.** The top six families are tied on the test period.
3. **Extra buzz features, tuning, ensembles and neural networks did not help** (the neural networks were clearly worse).
4. Probabilities run too high on the 2023+ films (Hit rate fell from 47% to 30%), so use the model for **ranking**, not for exact probabilities.
5. Hindi films are the weak spot.

## Not done (be honest about these)
- No poster, trailer-frame or plot-text features (needs data that is not in the files).
- YouTube statistics have no collection dates, so they cannot be proven pre-release.
- No causal claims: SHAP describes the model, not why films succeed.
- The labelled films are not a random sample of all films.
- Not deployed anywhere; the Streamlit app runs locally or in Docker.
- Not tested on every operating system; the Dockerfile has not been build-tested yet.
