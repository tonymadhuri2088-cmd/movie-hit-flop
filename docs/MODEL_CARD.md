# Model card: Indian Movie Hit / Flop Predictor

## Intended use
Teaching and research: explore what public pre-release information says about whether an Indian film earns at least twice its budget, and how reliable that signal is.
**Not intended for:** investment, financing or greenlighting decisions, or for judging any individual film.

## Model
- Task: binary classification, Hit = revenue / budget >= 2.0.
- Deployed models (Part B): a basic model (HistGradientBoosting, no promotion data) and a buzz-enhanced model (XGBoost, adds Wikipedia page views). Both are refit on all 854 labelled films for the app.
- Part A compares 19 model families on the same split.

## Data
- 854 labelled films (2015 to 2026): 578 development (before 2023) and 276 frozen test (2023 onward).
- Inputs: language, budget (US dollars), runtime, genres, release date and festivals, Wikipedia page-view rates, YouTube trailer statistics, cast/crew/studio track records (earlier films only).
- Sources: public metadata, Wikipedia and YouTube statistics. Check source terms before redistribution.

## Performance (frozen 2023+ test films)
| Model | Test AUC (95% CI) |
|---|---|
| Basic (HistGradientBoosting) | 0.735 (0.667 to 0.802) |
| Buzz-enhanced (XGBoost) | 0.776 (0.717 to 0.834) |
| Logistic regression reference | 0.759 (0.692 to 0.822) |
| Always predict the development Hit rate | 0.500 |

The three real models are statistically indistinguishable. In the 19-model course notebook, test AUC ranges from 0.66 to 0.79 and no model is convincingly better than logistic regression.

### Extended metrics on the frozen test set (reporting only)
Computed from the stored predictions of the one-time test scoring, at the threshold chosen on development data only. Nothing was refitted or selected. Brackets are bootstrap 95% intervals. Confusion cells are true Flop correctly called Flop / true Flop called Hit / true Hit called Flop / true Hit correctly called Hit.

| Model | Threshold | Balanced accuracy | Macro-F1 | MCC | Hit recall | Flop recall | PR-AUC | Brier | TN / FP / FN / TP |
|---|---|---|---|---|---|---|---|---|---|
| Basic (HGB) | 0.572 | 0.64 (0.59 to 0.70) | 0.65 (0.59 to 0.72) | 0.36 (0.24 to 0.48) | 0.34 (0.24 to 0.45) | 0.94 (0.90 to 0.97) | 0.56 (0.46 to 0.68) | 0.19 (0.17 to 0.21) | 182 / 12 / 54 / 28 |
| Buzz (XGBoost) | 0.621 | 0.64 (0.58 to 0.70) | 0.65 (0.58 to 0.72) | 0.36 (0.23 to 0.49) | 0.33 (0.22 to 0.44) | 0.94 (0.91 to 0.97) | 0.57 (0.47 to 0.69) | 0.17 (0.15 to 0.20) | 183 / 11 / 55 / 27 |
| Logistic regression | 0.517 | 0.63 (0.57 to 0.68) | 0.64 (0.57 to 0.70) | 0.31 (0.19 to 0.44) | 0.34 (0.24 to 0.44) | 0.91 (0.87 to 0.95) | 0.60 (0.50 to 0.71) | 0.18 (0.16 to 0.19) | 177 / 17 / 54 / 28 |
| Always the development Hit rate | 0.500 | 0.50 | 0.41 | 0.00 | 0.00 | 1.00 | 0.30 | 0.24 | 194 / 0 / 82 / 0 |

Reading: the models find about a third of Hits and are correct about 94% of the time when they say Flop. Accuracy of 76% should be read against the 70.3% obtained by always answering Flop. The three real models are not clearly different from each other on any of these metrics.

## Known limitations
- **Selection bias:** only films with public budget and revenue are labelled, and they are not a random sample.
- **Drift:** the Hit rate fell from 47% (before 2023) to 30% (2023 onward). Predicted probabilities run high for recent films, so use the model to rank films, not to read exact probabilities.
- **Uncertainty warnings (app):** a verdict within 5 points of the Hit line is a toss-up (right about 40% of the time on development films). A budget or runtime outside everything in the training films is flagged as extrapolation. Other "unusual input" notes are information only: on past films such inputs were not predicted worse. Probabilities were not recalibrated because no recalibration was supported as better.
- **Hindi films:** validation AUC about 0.56, close to guessing.
- **Money values:** US dollars as stored; gross versus net is not documented.
- **YouTube features:** collection dates are missing, so they cannot be proven to be pre-release.
- **Predictive, not causal:** explanations describe the model, not how to make a film succeed.
- **Small data:** 854 films means wide confidence intervals; differences of a few AUC points are noise.

## Ethical considerations
The model learns from past box-office patterns, including language and budget effects. It should not be used to disadvantage films, languages or filmmakers.
