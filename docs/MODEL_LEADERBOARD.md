# 19-model leaderboard (course notebook)

Sorted by validation AUC (expanding-window chronological cross-validation on 578 development films).
Test AUC is on the 276 frozen 2023+ films, with a 95% bootstrap interval. The test column is for reporting, not for choosing a model.
Because many models were tried, the best validation score is optimistic, and the intervals overlap heavily, so the top models are statistically tied.

| # | family | model | features | validation AUC | test AUC | test 95% CI |
|---|---|---|---|---|---|---|
| 1 | classic (extra) | XGBoost | D: base + Wikipedia + YouTube | 0.73 | 0.774 | 0.714 to 0.833 |
| 2 | classic (extra) | Gradient Boosting (sklearn) | B: base + Wikipedia | 0.721 | 0.771 | 0.712 to 0.829 |
| 3 | classic (extra) | Average ensemble (LR + RF + HGB) | B: base + Wikipedia | 0.717 | 0.791 | 0.729 to 0.849 |
| 4 | classic (main) | HistGradientBoosting | B: base + Wikipedia | 0.714 | 0.761 | 0.696 to 0.830 |
| 5 | classic (extra) | Stacking (LR on LR + RF + HGB) | B: base + Wikipedia | 0.714 | 0.788 | 0.727 to 0.847 |
| 6 | classic (extra) | CatBoost | D: base + Wikipedia + YouTube | 0.709 | 0.763 | 0.699 to 0.824 |
| 7 | classic (extra) | LightGBM | B: base + Wikipedia | 0.709 | 0.781 | 0.721 to 0.837 |
| 8 | classic (main) | Random forest | D: base + Wikipedia + YouTube | 0.706 | 0.756 | 0.687 to 0.818 |
| 9 | classic (extra) | Extra Trees | D: base + Wikipedia + YouTube | 0.706 | 0.723 | 0.652 to 0.790 |
| 10 | classic (extra) | AdaBoost | B: base + Wikipedia | 0.701 | 0.765 | 0.703 to 0.823 |
| 11 | classic (main) | Logistic regression | A: base | 0.69 | 0.759 | 0.692 to 0.822 |
| 12 | classic (extra) | Elastic-net logistic regression | A: base | 0.688 | 0.76 | 0.693 to 0.822 |
| 13 | classic (extra) | Linear discriminant analysis | A: base | 0.681 | 0.74 | 0.673 to 0.803 |
| 14 | neural network | MLP | A: base | 0.678 | 0.708 | 0.635 to 0.780 |
| 15 | classic (extra) | Gaussian Naive Bayes | A: base | 0.638 | 0.664 | 0.593 to 0.731 |
| 16 | neural network | 1D-CNN | B: base + Wikipedia | 0.622 | 0.736 | 0.667 to 0.803 |
| 17 | classic (extra) | k-Nearest Neighbours | A: base | 0.621 | 0.682 | 0.613 to 0.753 |
| 18 | neural network | GRU | B: base + Wikipedia | 0.614 | 0.751 | 0.684 to 0.818 |
| 19 | classic (extra) | SVM (RBF) | A: base | 0.591 | 0.73 | 0.658 to 0.800 |

Figures: [leaderboard](course_notebook_results/08c_all_models_leaderboard.png), [ROC/PR/confusion](course_notebook_results/05_test_roc_pr_confusion.png), [calibration](course_notebook_results/12_calibration.png), [SHAP](course_notebook_results/13_shap_summary.png), [shuffled-label check](course_notebook_results/09_shuffled_labels.png).
