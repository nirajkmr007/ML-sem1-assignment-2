"""
train_models.py
---------------
Trains and evaluates six classifiers on the wind-turbine gearbox
condition-monitoring feature set produced by ``feature_extraction.py``.

Task
====
6-class classification of the gearbox operating state from vibration alone:

    Healthy-Low  Healthy-Mid  Healthy-High
    Broken-Low   Broken-Mid   Broken-High

Models
======
1. Logistic Regression          4. Gaussian Naive Bayes
2. Decision Tree                5. Random Forest  (ensemble)
3. k-Nearest Neighbours         6. Support Vector Machine (RBF)

Every model is wrapped in a scikit-learn ``Pipeline`` together with a
``StandardScaler``. Persisting the scaler *inside* the model artefact means the
Streamlit app can score a freshly uploaded CSV without having to remember any
preprocessing state -- a common source of train/serve skew.

Metrics per model: Accuracy, ROC-AUC (macro one-vs-rest), Precision, Recall
and F1 (all macro-averaged so every state counts equally), and MCC.

The script also runs a sanity check on the *binary* health-only problem, to
document why the harder 6-class target was chosen.

Author: Niraj Kumar -- BITS WILP M.Tech (AI/ML), Machine Learning Assignment 2
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 17
TARGET_COLUMN = "gearbox_state"
CLASS_ORDER = [
    "Healthy-Low", "Healthy-Mid", "Healthy-High",
    "Broken-Low", "Broken-Mid", "Broken-High",
]
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
ARTEFACT_DIR = HERE / "saved"


# --------------------------------------------------------------------------
# Model zoo
# --------------------------------------------------------------------------

def build_model_zoo() -> dict[str, Pipeline]:
    """The six pipelines compared in this assignment.

    Hyper-parameters are deliberately modest and were chosen from the shape of
    the data (36 standardised features, ~2.7k training windows, six roughly
    balanced classes) rather than from an exhaustive grid search, so the
    comparison reflects the inductive bias of each algorithm rather than how
    much tuning effort each one received.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0, max_iter=5000, solver="lbfgs", random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(
                criterion="entropy", max_depth=10, min_samples_leaf=5,
                random_state=RANDOM_STATE)),
        ]),
        "kNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=7, weights="distance", metric="minkowski", p=2)),
        ]),
        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB(var_smoothing=1e-9)),
        ]),
        "Random Forest (Ensemble)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=400, max_depth=None, min_samples_leaf=2,
                max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=10.0, kernel="rbf", gamma="scale", probability=True,
                random_state=RANDOM_STATE)),
        ]),
    }


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """The six required metrics for one fitted pipeline.

    Precision / Recall / F1 are macro-averaged: with six operating states we
    care equally about every state, not just the most populated one. AUC uses
    the macro one-vs-rest formulation, which is the standard multi-class
    generalisation of the binary ROC-AUC.
    """
    y_pred = pipeline.predict(X)
    proba = pipeline.predict_proba(X)

    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(
            y, proba, multi_class="ovr", average="macro",
            labels=list(pipeline.classes_)),
        "Precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def binary_sanity_check(X_train, y_train, X_test, y_test) -> dict[str, float]:
    """Accuracy of each model on the *health-only* problem.

    Documents the claim in the README that the plain Healthy-vs-Broken task is
    saturated on this dataset and therefore cannot separate the six models.
    """
    to_health = lambda s: s.str.split("-").str[0]
    scores = {}
    for name, pipeline in build_model_zoo().items():
        pipeline.fit(X_train, to_health(y_train))
        scores[name] = round(
            accuracy_score(to_health(y_test), pipeline.predict(X_test)), 4)
    return scores


def artefact_filename(model_name: str) -> str:
    """'Random Forest (Ensemble)' -> 'random_forest_ensemble.joblib'."""
    slug = (model_name.lower()
            .replace("(", "").replace(")", "")
            .replace("-", " ").strip())
    return "_".join(slug.split()) + ".joblib"


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> None:
    train_df = pd.read_csv(PROJECT_ROOT / "train_data.csv")
    test_df = pd.read_csv(PROJECT_ROOT / "test_data.csv")

    feature_columns = [c for c in train_df.columns if c != TARGET_COLUMN]
    X_train, y_train = train_df[feature_columns], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[feature_columns], test_df[TARGET_COLUMN]

    print(f"Training windows : {len(X_train)}")
    print(f"Test windows     : {len(X_test)}")
    print(f"Features         : {len(feature_columns)}")
    print(f"Classes          : {sorted(y_train.unique())}\n")

    ARTEFACT_DIR.mkdir(parents=True, exist_ok=True)
    results, confusions = {}, {}

    for name, pipeline in build_model_zoo().items():
        pipeline.fit(X_train, y_train)
        results[name] = evaluate(pipeline, X_test, y_test)
        confusions[name] = confusion_matrix(
            y_test, pipeline.predict(X_test), labels=CLASS_ORDER).tolist()

        # compress=3 keeps the Random Forest artefact small enough to clone
        # comfortably and to fit the Streamlit Community Cloud free tier.
        joblib.dump(pipeline, ARTEFACT_DIR / artefact_filename(name), compress=3)
        print(f"{name:<26} " +
              "  ".join(f"{k}={v:.4f}" for k, v in results[name].items()))

    table = pd.DataFrame(results).T[METRIC_ORDER].round(4)
    table.index.name = "ML Model Name"
    winner = table["MCC"].idxmax()

    print("\nRunning binary health-only sanity check ...")
    binary_scores = binary_sanity_check(X_train, y_train, X_test, y_test)
    for name, acc in binary_scores.items():
        print(f"  {name:<26} binary accuracy = {acc:.4f}")

    (ARTEFACT_DIR / "metrics.json").write_text(json.dumps({
        "metrics": {k: {m: round(v, 4) for m, v in d.items()} for k, d in results.items()},
        "confusion_matrices": confusions,
        "class_order": CLASS_ORDER,
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "winner": winner,
        "binary_health_only_accuracy": binary_scores,
        "sklearn_version": __import__("sklearn").__version__,
    }, indent=2))

    table.to_csv(ARTEFACT_DIR / "comparison_table.csv")

    print("\n=== Comparison table (held-out test windows) ===")
    print(table.to_string())
    print(f"\nBest model by MCC: {winner}")
    print(f"Artefacts written to {ARTEFACT_DIR}")


if __name__ == "__main__":
    main()
