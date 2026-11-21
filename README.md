# Indian Movie Hit / Flop Prediction

![tests](https://github.com/tonymadhuri2088-cmd/movie-hit-flop/actions/workflows/ci.yml/badge.svg)

Can we tell, **before release**, whether an Indian film will earn at least twice its budget (**Hit** = revenue / budget >= 2)?

This repo answers that with a strict, leak-free evaluation and reports the result honestly: **moderately yes (test AUC about 0.76), and fancier models and extra "buzz" features do not change that.**

## Two parts

| Part | What it is | Where |
|---|---|---|
| **A. Course notebook** | One notebook that compares **19 model families** (classic, ensembles, neural networks) end to end. Runs in Google Colab in about 30 minutes. | `notebooks/01_course_notebook_19_models.ipynb`, results in `docs/course_notebook_results/` |
| **B. Research package** | The same question as a tested Python package: nested tuning, ensembles, calibration, SHAP, a frozen test set scored once, and a Streamlit app. | `hitflop/`, `tests/`, `app/`, `outputs/` |

The two parts use different feature sets and validation set-ups, so their exact numbers differ slightly (for example HistGradientBoosting test AUC 0.761 in A and 0.735 in B). The conclusions agree.

## Tech stack
| Area | Tools |
|---|---|
| Language | Python 3.11+ |
| Data | pandas, NumPy, SciPy |
| Classic models | scikit-learn (logistic regression, random forest, HistGradientBoosting, SVM, kNN, Naive Bayes, LDA, AdaBoost, Extra Trees), XGBoost, LightGBM, CatBoost |
| Neural networks (Part A) | PyTorch (MLP, GRU, 1D-CNN) |
| Tuning and explanation | Optuna (nested chronological tuning), SHAP, permutation importance |
| Features | `holidays` package for festival and holiday calendars |
| Statistics | paired bootstrap, Benjamini-Hochberg correction |
| App | Streamlit |
| Quality | pytest (59 tests), GitHub Actions CI, Docker |
| Notebook | Google Colab / Jupyter, matplotlib, seaborn |

## Data and split
- 854 films with both budget and revenue: **578 development (released before 2023-01-01)** and **276 frozen test (2023 onward)**. Seed 42.
- Features known before release: budget, language, genre, release timing and festivals, Wikipedia page-view rates, YouTube trailer statistics, cast/crew/studio track records (only from films released at least 120 days earlier).
- The test set is scored once. In Part B a lock file (`outputs/final/TEST_SCORED.lock`) refuses a second run.

## Headline results (Part A, 19 models)
Test AUC on the 276 frozen films. Chance is 0.50.

| Model | Features | Validation AUC | Test AUC (95% CI) |
|---|---|---|---|
| XGBoost | base + Wikipedia + YouTube | 0.730 | 0.774 (0.714 to 0.833) |
| HistGradientBoosting (chosen by cross-validation) | base + Wikipedia | 0.714 | 0.761 (0.696 to 0.830) |
| Random forest | base + Wikipedia + YouTube | 0.706 | 0.756 (0.687 to 0.818) |
| Logistic regression (simple baseline) | base | 0.690 | 0.759 (0.692 to 0.822) |
| MLP / 1D-CNN / GRU (neural networks) | various | 0.61 to 0.68 | 0.71 / 0.74 / 0.75 |

Full 19-row table: [`docs/MODEL_LEADERBOARD.md`](docs/MODEL_LEADERBOARD.md).

What this means:
1. **Hits can be ranked better than chance** (AUC about 0.76), but only moderately. At its chosen threshold the main model finds about 35% of Hits and 91% of Flops.
2. **No model is convincingly better than logistic regression.** All paired-bootstrap intervals include 0 on the held-out period. The top six model families are statistically tied.
3. **Neural networks are worse**, as expected for 578 training films.
4. **Pre-release buzz (Wikipedia, YouTube) adds little.** In Part B, 0 of 48 feature comparisons were convincing, and Optuna tuning and ensembles did not help either.
5. **The ceiling is the data, not the algorithm.** Budget and language dominate.

More detail and caveats are in [`docs/WHAT_IS_DONE.md`](docs/WHAT_IS_DONE.md). Ideas for next steps are in [`docs/IMPROVEMENTS.md`](docs/IMPROVEMENTS.md).

## App
An interactive Streamlit app with a clean front end: a hero banner with key numbers, a **Predict** tab (film details in, chance of a Hit, a colour-coded verdict, per-input SHAP bars and honest warnings), a **Model comparison** tab, a **Research dashboard** (data quality, robustness, experiment log) and an **About & limits** tab.

![Predict page with SHAP explanation](docs/screenshots/app_predict.png)

More screenshots: [model comparison](docs/screenshots/app_model_comparison.png), [research dashboard](docs/screenshots/app_dashboard.png).

Run it locally with `streamlit run app/streamlit_app.py`, or with Docker (below).

## Quick start
```bash
git clone https://github.com/tonymadhuri2088-cmd/movie-hit-flop.git
cd movie-hit-flop
pip install -r requirements.txt
python -m pytest -q tests          # 59 tests: cohort, leakage, folds, metrics, calibration, app
streamlit run app/streamlit_app.py # Predict / Model comparison / Research dashboard / About & limits
```
Run Part A: open `notebooks/01_course_notebook_19_models.ipynb` in Google Colab and choose Runtime > Run all (about 30 minutes). The first cell downloads the two data files automatically; if that fails, Colab shows an upload button, so upload `data/movies_full.csv` and `data/yt_features.csv` from this repo. Locally, also run `pip install -r requirements-notebook.txt`.

Run with Docker (the Dockerfile is written but has not been build-tested yet):
```bash
docker build -t movie-hit-flop .
docker run -p 8501:8501 movie-hit-flop   # then open http://localhost:8501
```

Re-create Part B results:
```bash
python -m hitflop.pipeline_phase1; python -m hitflop.phase3; python -m hitflop.phase45
python -m hitflop.final            # scores the frozen test set ONCE (lock file)
python -m hitflop.artifacts; python -m hitflop.phase6
```

## Repository map
```
hitflop/        package: data, features, history (leak-free records), models, validation, metrics,
                calibration, tuning (nested Optuna), ensembles, explain (SHAP), predict, final
tests/          59 tests, including leakage tests that flip future labels and check earlier features do not move
app/            Streamlit app
notebooks/      01 = 19-model course notebook, 02 = upgrade experiments
experiments/    append-only experiment log (every experiment, settings and result)
outputs/        Part B tables and figures
docs/           what is done, improvements, 19-model leaderboard, figures, app screenshots
data/           labelled movies, Wikipedia infobox records, YouTube features
```

## Method notes
- **Validation:** expanding-window walk-forward (chronological), never random shuffling. Tuning uses a nested inner chronological split.
- **Comparisons:** paired bootstrap on identical films with Benjamini-Hochberg correction. "Convincing" = interval above 0 and adjusted q < 0.05.
- **Leakage controls:** track records use only earlier films; label-flip perturbation tests prove it. The Hit threshold (2.0) is fixed; 1.5 and 2.5 are sensitivity checks only.
- **Calibration:** calibrators fitted on earlier folds made probabilities worse because the Hit rate drifts (47% before 2023, 30% after), so they are not used. Predictions run high.

## Limits
- Predictive, not causal. SHAP explains the model, not the real world.
- Only 854 of 4,273 films have budget and revenue, and those are not a random sample (selection-model AUC 0.86).
- Money is in US dollars as stored; gross versus net is undocumented.
- YouTube collection dates are missing, so YouTube features cannot be proven to be pre-release.
- Hindi is the weakest language (validation AUC about 0.56).
- Library versions matter: scikit-learn and the `holidays` package change results slightly (AUC within about 0.02, same conclusions).

## Data note
The data files were compiled from public sources (movie metadata, Wikipedia page views and infoboxes, YouTube trailer statistics). Check each source's terms before reusing or redistributing the data.

## Licence
MIT. See [LICENSE](LICENSE).
