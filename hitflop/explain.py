"""SHAP explanations. IMPORTANT WORDING: SHAP describes how the *model* uses each feature to predict. It is predictive attribution, not causal proof:
'a high Wikipedia rate pushes the predicted Hit probability up' does not mean that raising page views would make the film a Hit."""
import numpy as np
import pandas as pd
import shap


def transformed(pipe, X):
    """Run the fitted preprocessing step and return (matrix, feature names)."""
    prep = pipe.named_steps["prep"]
    Z = prep.transform(X)
    import re
    names = [re.sub(r"^(num|cat|genre|blk\d+)__", "", n) for n in prep.get_feature_names_out()]
    return np.asarray(Z, float), names


def _explainer(model, Zbg):
    cls = type(model).__name__
    if cls in ("XGBClassifier", "LGBMClassifier", "CatBoostClassifier", "RandomForestClassifier"):
        return shap.TreeExplainer(model)
    if cls == "LogisticRegression":
        return shap.LinearExplainer(model, Zbg)
    return shap.Explainer(lambda z: model.predict_proba(z)[:, 1], Zbg, max_samples=len(Zbg))       # slow generic fallback (e.g. HistGradientBoosting)


def shap_values(pipe, X_background, X_explain):
    """SHAP values (log-odds scale for tree/linear models) for the rows of X_explain. Returns (values, data, feature names)."""
    Zbg, names = transformed(pipe, X_background)
    Z, _ = transformed(pipe, X_explain)
    model = pipe.named_steps["model"]
    ex = _explainer(model, Zbg)
    sv = ex(Z) if type(model).__name__ in ("LogisticRegression",) or ex.__class__.__name__ == "Permutation" else ex.shap_values(Z)
    vals = sv.values if hasattr(sv, "values") else sv
    if isinstance(vals, list):
        vals = vals[1]
    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    return vals, Z, names


def global_importance(vals, names):
    return pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(vals).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def local_explanation(vals_row, data_row, names, top=8):
    """Top positive and negative contributions for ONE movie."""
    d = pd.DataFrame({"feature": names, "value": data_row, "shap": vals_row})
    d["abs"] = d.shap.abs()
    d = d.sort_values("abs", ascending=False).head(top).drop(columns="abs")
    return d.reset_index(drop=True)


def dependence_table(vals, Z, names, feature):
    j = names.index(feature)
    return pd.DataFrame({"value": Z[:, j], "shap": vals[:, j]})


def group_importance(imp, groups):
    """Sum mean |SHAP| per named group (prefix match on the transformed feature names)."""
    out = {}
    for g, prefixes in groups.items():
        out[g] = float(imp[imp.feature.apply(lambda f: any(f == p or f.startswith(p) for p in prefixes))].mean_abs_shap.sum())
    return pd.Series(out).sort_values(ascending=False)
