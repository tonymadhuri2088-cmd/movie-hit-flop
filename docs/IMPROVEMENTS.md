# What could be improved next

Ordered by how likely each is to help, based on what the experiments showed. The main finding, confirmed by the research upgrade, is that **the data, not the algorithm, limits accuracy**: imbalance methods, rolling windows, recalibration, extra feature blocks, more models, more tuning and more ensembles were all tried and none was supported as better.

## 1. Highest value: better and more data
- **More labelled films.** Only 854 of 4,273 films have budget and revenue. Add budget and box-office figures from more sources and re-check the selection bias.
- **More Hindi films with informative features.** Inside Hindi (237 validated films) the models are not clearly better than guessing, so this is where new information is needed most.
- **Dated YouTube snapshots.** Collect trailer views, likes and comments with a timestamp (for example 30, 14 and 7 days before release) so they are provably pre-release.
- **Plot text and posters.** Sentence embeddings of the plot summary, and poster or trailer-frame features. This is where a CNN or transformer could finally have something to learn from.

## 2. Better question framing
- Predict **ROI classes** (Flop / Average / Hit / Blockbuster) or the log-ROI itself, then pick a threshold for the decision. A film at ROI 1.9 and one at 2.1 are almost the same, but the label differs.
- **Expected-value decisions:** score a film by probability times upside minus downside, not by Hit/Flop alone.

## 3. Open questions the upgrade could not settle
- **Recency recalibration.** An intercept shift fitted on the two most recent development blocks looked helpful on the frozen test films (reporting only) but was not supported on development folds. Re-validate it on films released after 2026 once they have box-office results.
- **Does it transfer to a new language?** The leave-one-language-out test shows no clear loss, but its intervals are about 0.2 wide. A second regional-language dataset would answer it.
- **Conformal prediction intervals** instead of rule-based warnings. Only the close-call warning is currently backed by evidence.
- **A hierarchical model by language** could share strength across languages without pooling them blindly.

## 4. Engineering
- A small FastAPI service for the model (the app and its Docker image already exist).
- Experiment tracking (MLflow) and data versioning (DVC).
- Mutation testing or property-based tests for the feature builders.

## 5. Done since the first version
Model card, `CITATION.cff`, contributing guide, issue and pull-request templates, `ruff` with `pre-commit`, coverage and a Docker build checked in CI, a deployed Streamlit app, an uncertainty warning backed by evidence, and the full upgrade log (`docs/UPGRADE_LOG.md`).

## 6. Not worth more effort (evidence says no)
- More model families or more Optuna trials: 0 of 10 tuned models beat defaults, and the top six families are tied.
- Class weights, Balanced Random Forest, SMOTE-NC or other threshold rules: all 36 method-versus-plain comparisons are "no clear difference", and maximum-F1 thresholding is clearly worse.
- Rolling windows or time-decay weights: 17 of 40 comparisons clearly worse, none better.
- Platt, isotonic or intercept-shift recalibration of the deployed models: none supported as better; isotonic clearly worse.
- More hand-made Wikipedia or YouTube ratios, or extra track-record, calendar and competition blocks: 0 of 48 comparisons were convincing, and the Phase 5 ablation finds no clear help.
