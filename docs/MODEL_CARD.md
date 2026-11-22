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

## Known limitations
- **Selection bias:** only films with public budget and revenue are labelled, and they are not a random sample.
- **Drift:** the Hit rate fell from 47% (before 2023) to 30% (2023 onward). Predicted probabilities run high for recent films, so use the model to rank films, not to read exact probabilities.
- **Hindi films:** validation AUC about 0.56, close to guessing.
- **Money values:** US dollars as stored; gross versus net is not documented.
- **YouTube features:** collection dates are missing, so they cannot be proven to be pre-release.
- **Predictive, not causal:** explanations describe the model, not how to make a film succeed.
- **Small data:** 854 films means wide confidence intervals; differences of a few AUC points are noise.

## Ethical considerations
The model learns from past box-office patterns, including language and budget effects. It should not be used to disadvantage films, languages or filmmakers.
