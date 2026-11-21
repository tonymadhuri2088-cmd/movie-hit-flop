# What could be improved next

Ordered by how likely each is to help, based on what the experiments showed. The main finding is that **the data, not the algorithm, limits accuracy**. More models, more tuning and more ensembles were tried and did not help, so they are at the bottom.

## 1. Highest value: better and more data
- **More labelled films.** Only 854 of 4,273 films have budget and revenue. Add budget and box-office figures from more sources and re-check the selection bias.
- **Dated YouTube snapshots.** Collect trailer views, likes and comments with a timestamp (for example 30, 14 and 7 days before release) so they are provably pre-release.
- **Plot text and posters.** Sentence embeddings of the plot summary, and poster or trailer-frame features. This is where a CNN or transformer could finally have something to learn from.

## 2. Better question framing
- Predict **ROI classes** (Flop / Average / Hit / Blockbuster) or the log-ROI itself, then pick a threshold for the decision. A film at ROI 1.9 and one at 2.1 are almost the same, but the label differs.
- **Expected-value decisions:** score a film by probability times upside minus downside, not by Hit/Flop alone.

## 3. Weak spots
- Hindi has the lowest validation AUC (about 0.56). Try language-specific models or a hierarchical model, or collect more Hindi features.
- **Drift:** the Hit rate fell from 47% to 30%. Re-fit on a rolling window and track the base rate over time.
- Report **uncertainty** in the app (conformal prediction intervals) instead of a single probability.

## 4. Engineering polish
- A small FastAPI service for the model (a Dockerfile for the app already exists).
- Experiment tracking (MLflow) and data versioning (DVC).
- A model card (intended use, data, limits) and a `CITATION.cff`.
- Type hints, `ruff` or `black`, and `pre-commit`.
- Deploy the Streamlit app (for example Streamlit Community Cloud) and link it in the README.

## 5. Not worth more effort (evidence says no)
- More model families or more Optuna trials: 0 of 10 tuned models beat defaults, and the top six families are tied.
- More hand-made Wikipedia or YouTube ratios: 0 of 48 comparisons were convincing.
- Calibrating on earlier folds: it made probabilities worse because of drift.
