"""Train and compare baseline regressors for protein solubility."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from .data_utils import ROOT, load_config, load_training_data

def metrics(y_true, y_pred):
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None}

def main():
    cfg = load_config()
    X, y, summary = load_training_data()
    if len(X) < 10:
        raise SystemExit(f"Only {len(X)} usable rows. At least 10 are recommended for a train/test split.")
    reports = ROOT / "reports"; models_dir = ROOT / "models"
    reports.mkdir(exist_ok=True); models_dir.mkdir(exist_ok=True)
    (reports / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], random_state=cfg["random_state"])
    candidates = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Pipeline([("imputer", SimpleImputer(strategy="median")),
                           ("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "random_forest": Pipeline([("imputer", SimpleImputer(strategy="median")),
                                   ("model", RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                                                  random_state=cfg["random_state"], n_jobs=-1))]),
        "extra_trees": Pipeline([("imputer", SimpleImputer(strategy="median")),
                                 ("model", ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2,
                                                              random_state=cfg["random_state"], n_jobs=-1))])
    }
    rows, fitted = [], {}
    for name, estimator in candidates.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(X_train, y_train)
        pred = estimator.predict(X_test)
        row = {"model": name, **metrics(y_test, pred)}
        rows.append(row); fitted[name] = estimator
        print(f"{name:15s} MAE={row['mae']:.4f} RMSE={row['rmse']:.4f} R2={row['r2']:.4f}")
    comparison = pd.DataFrame(rows).sort_values("mae")
    comparison.to_csv(reports / "model_comparison.csv", index=False)
    best_name = comparison.iloc[0]["model"]
    best_model = fitted[best_name]
    predictions = best_model.predict(X_test)
    pd.DataFrame({"actual": y_test.to_numpy(), "predicted": predictions},
                 index=y_test.index).rename_axis("row_index").to_csv(reports / "test_predictions.csv")
    final_metrics = {"selected_model": best_name, "selection_metric": "lowest test-set MAE",
                     "test_metrics": metrics(y_test, predictions), "dataset": summary,
                     "note": "Metrics are from a single random holdout split; external generalization is not guaranteed."}
    (reports / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    bundle = {"model": best_model, "feature_columns": list(X.columns), "target_column": summary["target_column"],
              "sequence_based": summary["sequence_based"], "selected_model": best_name}
    joblib.dump(bundle, models_dir / "protein_solubility_model.joblib")
    print(f"\\nSelected: {best_name}. Reports saved under {reports}; model saved under {models_dir}.")
    print("Reminder: check dataset provenance and whether the split is suitable for your scientific question.")

if __name__ == "__main__":
    main()
