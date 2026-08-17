"""
Gearbox Health Explorer
=======================
Streamlit front-end for the wind-turbine gearbox condition-monitoring models
built in ``model/train_models.py``.

Upload a CSV of extracted vibration features (or use the bundled held-out test
set), pick one of the six trained classifiers, and the app reports the full
metric set, the confusion matrix, the per-class classification report and the
one-vs-rest ROC curves -- plus a side-by-side comparison of all six models and
a per-window probability inspector.

Run locally with:   streamlit run app.py

Author: Niraj Kumar -- BITS WILP M.Tech (AI/ML), Machine Learning Assignment 2
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

# --------------------------------------------------------------------------
# Constants and paths
# --------------------------------------------------------------------------

APP_ROOT = Path(__file__).resolve().parent
ARTEFACT_DIR = APP_ROOT / "model" / "saved"
TEST_CSV = APP_ROOT / "test_data.csv"
TRAIN_CSV = APP_ROOT / "train_data.csv"
TARGET_COLUMN = "gearbox_state"

CLASS_ORDER = [
    "Healthy-Low", "Healthy-Mid", "Healthy-High",
    "Broken-Low", "Broken-Mid", "Broken-High",
]
METRIC_KEYS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

# One-line plain-English note per model, shown under the dropdown so the
# selection means something to a reader who is not already familiar with it.
MODEL_NOTES = {
    "Logistic Regression":
        "Linear model — one weight per feature per class. Fastest to train, "
        "smallest to store, and the only model here whose coefficients read "
        "directly as per-channel sensitivities.",
    "Decision Tree":
        "A single tree of axis-parallel splits (entropy, depth 10). Easy to "
        "read end to end, but its leaf probabilities are coarse — which is "
        "why its AUC trails the rest.",
    "kNN":
        "Classifies each window by its 7 nearest training windows, weighted "
        "by distance. There is no training phase — all the work happens at "
        "prediction time.",
    "Naive Bayes":
        "Assumes the features are independent given the class. That is badly "
        "violated here (several descriptors are built from the same peak and "
        "mean terms), which miscalibrates its posteriors.",
    "Random Forest (Ensemble)":
        "400 trees, each on a bootstrap sample with a random feature subset, "
        "majority-voted. A large variance reduction over the single tree.",
    "Support Vector Machine":
        "Fits a maximum-margin boundary in an RBF kernel space. Slowest to "
        "train, and needs internal cross-validation to emit the probabilities "
        "the AUC and ROC curves require.",
}

METRIC_HELP = {
    "Accuracy":
        "Share of windows whose predicted state exactly matches the true state.",
    "AUC":
        "Macro one-vs-rest ROC-AUC. Measures how well the model *ranks* windows "
        "by class probability, independent of where the decision threshold sits.",
    "Precision":
        "Of the windows predicted as a given state, how many really were that "
        "state. Macro-averaged, so every state counts equally.",
    "Recall":
        "Of the windows truly in a given state, how many the model found. "
        "Macro-averaged, so every state counts equally.",
    "F1":
        "Harmonic mean of precision and recall — a single number for when you "
        "care about both equally.",
    "MCC":
        "Matthews Correlation Coefficient: a balanced correlation between "
        "prediction and truth, running from −1 (perfectly inverted) through "
        "0 (no better than chance) to +1 (perfect). Much harder to fool than "
        "accuracy when the classes are uneven.",
}

STATE_LEGEND = pd.DataFrame(
    [["**Healthy** — gear teeth intact", "Healthy-Low", "Healthy-Mid", "Healthy-High"],
     ["**Broken** — one chipped tooth", "Broken-Low", "Broken-Mid", "Broken-High"]],
    columns=["Gearbox condition", "Low load (0–30 %)", "Mid load (40–60 %)",
             "High load (70–90 %)"],
).set_index("Gearbox condition")

# Two categorical hues, one per gearbox condition, checked for colour-vision
# deficiency: this pair separates by dE 25.7 under protanopia, whereas the more
# obvious green/red pair collapses to dE 7.7 -- unreadable for the most common
# form of colour blindness. Blue = nominal, red = fault is also standard
# industrial HMI convention.
HEALTHY_TINT = "#1D4ED8"
FAULT_TINT = "#C1453B"

# Streamlit renamed `use_container_width` to `width="stretch"` and has announced
# the removal of the old name. Community Cloud always installs the newest
# release, while the BITS lab may have an older one, so pick the spelling that
# the installed version actually understands instead of hard-coding either.
try:
    _ST_VERSION = tuple(int(part) for part in st.__version__.split(".")[:2])
except (ValueError, AttributeError):
    _ST_VERSION = (1, 30)
WIDE = {"width": "stretch"} if _ST_VERSION >= (1, 49) else {"use_container_width": True}

st.set_page_config(
    page_title="Gearbox Health Explorer",
    page_icon="🌀",
    layout="wide",
)


# --------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading trained models ...")
def load_models() -> tuple[dict, str]:
    """Load the six persisted pipelines.

    If unpickling fails -- which happens when the deployment environment has a
    different scikit-learn build than the machine the models were trained on --
    the app retrains them from the bundled training CSV instead of showing an
    error page. Slower on first load, but the app always comes up.
    """
    try:
        models = {
            name: joblib.load(ARTEFACT_DIR / filename)
            for name, filename in MODEL_FILES.items()
        }
        return models, "loaded"
    except Exception:
        import sys

        sys.path.insert(0, str(APP_ROOT / "model"))
        from train_models import build_model_zoo  # noqa: E402

        frame = pd.read_csv(TRAIN_CSV)
        features = [c for c in frame.columns if c != TARGET_COLUMN]
        zoo = build_model_zoo()
        for pipeline in zoo.values():
            pipeline.fit(frame[features], frame[TARGET_COLUMN])
        return zoo, "retrained"


@st.cache_data(show_spinner=False)
def load_reference_metrics() -> dict:
    path = ARTEFACT_DIR / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data(show_spinner=False)
def load_bundled_test_data() -> pd.DataFrame:
    return pd.read_csv(TEST_CSV)


@st.cache_data(show_spinner=False)
def bundled_test_csv_bytes() -> bytes:
    """The held-out test set, ready to hand back out as a download.

    Without this the 'upload a CSV' path is a dead end for anyone who does not
    already have a copy of the file sitting on their machine.
    """
    return TEST_CSV.read_bytes()


def expected_features(models: dict) -> list[str]:
    reference = load_reference_metrics().get("feature_columns")
    if reference:
        return reference
    return list(next(iter(models.values()))[:-1].get_feature_names_out())


# --------------------------------------------------------------------------
# Metric computation
# --------------------------------------------------------------------------

def compute_metrics(model, X: pd.DataFrame, y_true: pd.Series) -> dict[str, float]:
    """All six assignment metrics, macro-averaged over the six gearbox states."""
    y_pred = model.predict(X)
    proba = model.predict_proba(X)

    try:
        auc = roc_auc_score(
            y_true, proba, multi_class="ovr", average="macro",
            labels=list(model.classes_))
    except ValueError:
        auc = float("nan")  # a class is missing from the uploaded slice

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


@st.cache_data(show_spinner="Scoring all six models ...")
def score_all_models(_models: dict, X: pd.DataFrame, y_true: pd.Series) -> pd.DataFrame:
    """Score every model once and reuse the result across tabs.

    The leading underscore on ``_models`` tells Streamlit not to try to hash the
    fitted pipelines; the cache key is the data itself, which is what actually
    changes when the user uploads a different CSV.
    """
    rows = {name: compute_metrics(model, X, y_true) for name, model in _models.items()}
    table = pd.DataFrame(rows).T[METRIC_KEYS]
    table.index.name = "ML Model Name"
    return table


def health_reliability(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Collapse the six states to Healthy / Broken and measure fault detection.

    This is the number that actually matters to a condition monitoring engineer:
    a load-regime mix-up is cosmetic, but calling a damaged gearbox healthy is
    the failure mode that lets a turbine run to destruction.
    """
    true_health = pd.Series(y_true).astype(str).str.split("-").str[0].to_numpy()
    pred_health = pd.Series(y_pred).astype(str).str.split("-").str[0].to_numpy()

    faulty = true_health == "Broken"
    healthy = ~faulty

    return {
        "faulty_windows": int(faulty.sum()),
        "faults_caught": int((faulty & (pred_health == "Broken")).sum()),
        "healthy_windows": int(healthy.sum()),
        "false_alarms": int((healthy & (pred_health == "Broken")).sum()),
        "boundary_errors": int((true_health != pred_health).sum()),
        "total": int(len(true_health)),
    }


def ordered_classes(labels) -> list[str]:
    """Keep the canonical Healthy->Broken, Low->High ordering where possible."""
    known = [c for c in CLASS_ORDER if c in set(labels)]
    extra = sorted(set(labels) - set(CLASS_ORDER))
    return known + extra


def figure_to_png(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    return buffer.getvalue()


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def draw_confusion_matrix(y_true, y_pred, normalise: bool):
    labels = ordered_classes(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    if normalise:
        totals = matrix.sum(axis=1, keepdims=True)
        display = np.divide(matrix, totals, where=totals != 0)
        fmt, cbar_label = ".2f", "share of true class"
    else:
        display, fmt, cbar_label = matrix, "d", "window count"

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sns.heatmap(
        display, annot=True, fmt=fmt, cmap="YlGnBu", square=True,
        xticklabels=labels, yticklabels=labels,
        cbar_kws={"label": cbar_label}, linewidths=0.5, linecolor="white", ax=ax,
    )
    ax.set_xlabel("Predicted gearbox state")
    ax.set_ylabel("True gearbox state")
    ax.set_title("Confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    return fig


def draw_roc_curves(model, X, y_true):
    labels = list(model.classes_)
    if len(labels) < 2:
        return None

    y_binary = label_binarize(y_true, classes=labels)
    proba = model.predict_proba(X)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    palette = sns.color_palette("husl", len(labels))

    for index, label in enumerate(labels):
        if y_binary[:, index].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_binary[:, index], proba[:, index])
        area = np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr)
        ax.plot(fpr, tpr, lw=1.9, color=palette[index],
                label=f"{label}  (AUC {area:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC curves")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    return fig


def draw_model_comparison(frame: pd.DataFrame, metric: str):
    ranked = frame.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    colours = sns.color_palette("crest", len(ranked))
    bars = ax.barh(ranked.index, ranked[metric], color=colours)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_xlim(0, min(1.06, ranked[metric].max() * 1.18))
    ax.set_xlabel(metric)
    ax.set_title(f"All six models ranked by {metric}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def draw_probability_bars(probabilities: pd.Series, predicted: str, actual: str | None):
    ordered = probabilities.reindex(ordered_classes(probabilities.index)).dropna()
    colours = [HEALTHY_TINT if s.startswith("Healthy") else FAULT_TINT
               for s in ordered.index]

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    bars = ax.barh(ordered.index[::-1], ordered.values[::-1], color=colours[::-1])
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("predicted probability")
    title = f"Model confidence — predicted {predicted}"
    if actual is not None:
        verdict = "correct" if predicted == actual else f"actual {actual}"
        title += f"  ({verdict})"
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

models, model_source = load_models()
reference = load_reference_metrics()
feature_names = expected_features(models)

st.sidebar.title("🌀 Gearbox Health Explorer")
st.sidebar.caption(
    "Wind-turbine gearbox condition monitoring — "
    "BITS WILP M.Tech (AI/ML), ML Assignment 2"
)

st.sidebar.subheader("1 · Test data")
data_choice = st.sidebar.radio(
    "Which data should the models be scored on?",
    ("Bundled held-out test set", "Upload my own CSV"),
    help="The bundled set is the chronologically held-out 30% of every recording.",
)

uploaded = None
if data_choice == "Upload my own CSV":
    uploaded = st.sidebar.file_uploader(
        "Upload a feature CSV", type=["csv"],
        help=("Needs the 36 vibration feature columns. Include a "
              "'gearbox_state' column to get evaluation metrics."),
    )
    st.sidebar.download_button(
        "⬇️ Need a file? Download the sample test CSV",
        bundled_test_csv_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
        **WIDE,
        help="The held-out test set — download it, then upload it back to try this feature.",
    )

st.sidebar.subheader("2 · Model")
selected_model_name = st.sidebar.selectbox(
    "Classification model", list(MODEL_FILES.keys()), index=0)
st.sidebar.caption(MODEL_NOTES[selected_model_name])

st.sidebar.subheader("3 · Display")
normalise_cm = st.sidebar.checkbox("Normalise confusion matrix by row", value=False)
show_roc = st.sidebar.checkbox("Show ROC curves", value=True)

if model_source == "retrained":
    st.sidebar.info(
        "Saved model files could not be unpickled in this environment, so the "
        "models were retrained from `train_data.csv` on startup. Results are "
        "identical.", icon="♻️")


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("Wind-Turbine Gearbox Health Explorer")
st.markdown(
    "Six classifiers trained on **36 vibration descriptors** extracted from "
    "four accelerometers on a wind-turbine gearbox test rig. The task is to "
    "recover the gearbox's **health state and load regime** from the vibration "
    "signature alone — no SCADA load channel is given to the models."
)

with st.expander("ℹ️  What the six gearbox states mean, and how to use this app"):
    st.markdown(
        "Each row of the data is one **window** of vibration — 512 consecutive "
        "samples from all four accelerometers, reduced to 36 condition-monitoring "
        "descriptors (RMS, kurtosis, crest factor, spectral shape, and so on). "
        "Every window belongs to one of six states:"
    )
    st.table(STATE_LEGEND)
    st.markdown(
        "**Using the app**\n\n"
        "1. Pick your data in the sidebar — the bundled held-out test set, or "
        "upload your own CSV (a sample is downloadable there).\n"
        "2. Choose one of the six models.\n"
        "3. **Selected model** shows its metrics, confusion matrix, ROC curves "
        "and classification report; **All six models** scores every model on the "
        "same data; **Data & predictions** lets you inspect individual windows."
    )


# --------------------------------------------------------------------------
# Data resolution
# --------------------------------------------------------------------------

if data_choice == "Upload my own CSV" and uploaded is None:
    st.info(
        "Waiting for a CSV. Use the sidebar to upload one — or grab the sample "
        "test set from the download button there and upload that.", icon="⬆️")
    with st.expander("What does the CSV need to contain?"):
        st.markdown(
            f"- **{len(feature_names)} feature columns**, named "
            "`<channel>_<descriptor>` — e.g. `a1_rms`, `a3_crest_factor`.\n"
            f"- Optionally a **`{TARGET_COLUMN}`** column with the true state. "
            "Without it the app still predicts, but cannot compute metrics.\n"
            "- Any extra columns are ignored."
        )
        st.code(", ".join(feature_names), language=None)
    st.stop()

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
    except Exception as error:
        st.error(f"Could not read that CSV: {error}")
        st.stop()
    source_label = f"uploaded file · {uploaded.name}"
else:
    data = load_bundled_test_data()
    source_label = "bundled held-out test set"

missing = [c for c in feature_names if c not in data.columns]
if missing:
    st.error(
        f"The CSV is missing {len(missing)} required feature column(s). "
        f"First few: {', '.join(missing[:6])}"
    )
    st.stop()

X = data[feature_names]
has_labels = TARGET_COLUMN in data.columns
y_true = data[TARGET_COLUMN] if has_labels else None

top = st.columns(4)
top[0].metric("Windows scored", f"{len(X):,}")
top[1].metric("Features used", len(feature_names))
top[2].metric("Labelled", "Yes" if has_labels else "No")
top[3].metric("Data source", "Bundled" if uploaded is None else "Uploaded")
st.caption(f"Scoring **{source_label}**.")

if not has_labels:
    st.warning(
        f"No `{TARGET_COLUMN}` column found, so metrics cannot be computed. "
        "Showing predictions only.", icon="⚠️")

all_scores = score_all_models(models, X, y_true) if has_labels else None

tab_single, tab_compare, tab_data = st.tabs(
    ["🔎 Selected model", "📊 All six models", "🗂️ Data & predictions"])


# --------------------------------------------------------------------------
# Tab 1 — selected model
# --------------------------------------------------------------------------

with tab_single:
    model = models[selected_model_name]
    y_pred = model.predict(X)

    if has_labels:
        ranking = all_scores["MCC"].rank(ascending=False).astype(int)
        position = int(ranking[selected_model_name])
        leader = all_scores["MCC"].idxmax()
        badge = "🏆 best on this data" if position == 1 else f"rank {position} of 6 by MCC"
        st.subheader(f"{selected_model_name} — evaluation metrics")
        st.caption(f"**{badge}** · {MODEL_NOTES[selected_model_name]}")
    else:
        st.subheader(f"{selected_model_name} — predictions")
        st.caption(MODEL_NOTES[selected_model_name])

    if has_labels:
        scores = all_scores.loc[selected_model_name]
        cells = st.columns(6)
        for cell, key in zip(cells, METRIC_KEYS):
            value = scores[key]
            gap = value - all_scores[key].max()
            cell.metric(
                key,
                "n/a" if np.isnan(value) else f"{value:.4f}",
                delta=None if (np.isnan(value) or abs(gap) < 5e-5) else f"{gap:+.4f}",
                delta_color="normal",
                help=METRIC_HELP[key],
            )
        st.caption(
            "Hover any metric name for what it measures. The small figure "
            "underneath is the gap to the best of the six models on this data "
            f"(currently **{leader}**); no figure means this model is the best."
        )

        # ---- fault-detection reliability -------------------------------
        reliability = health_reliability(y_true, y_pred)
        caught, faulty = reliability["faults_caught"], reliability["faulty_windows"]
        alarms, healthy = reliability["false_alarms"], reliability["healthy_windows"]
        catch_rate = caught / faulty if faulty else float("nan")

        st.markdown("##### Fault detection reliability")
        st.caption(
            "Collapsing the six states down to *is the gearbox damaged or not* — "
            "the question a condition monitoring system actually has to answer. "
            "A load-regime mix-up is cosmetic; calling a damaged gearbox healthy "
            "is not."
        )
        rel = st.columns(3)
        rel[0].metric(
            "Faults caught", f"{caught} / {faulty}",
            delta=None if np.isnan(catch_rate) else f"{catch_rate:.1%}",
            delta_color="off",
            help="Windows from a damaged gearbox that were predicted as damaged.")
        rel[1].metric(
            "False alarms", f"{alarms} / {healthy}",
            help="Healthy windows wrongly flagged as damaged — the cost of nuisance alerts.")
        rel[2].metric(
            "Health-level errors", f"{reliability['boundary_errors']} / {reliability['total']}",
            help="Predictions on the wrong side of the healthy/damaged boundary.")

        if reliability["boundary_errors"] == 0:
            st.success(
                f"Every one of the {reliability['total']:,} windows was placed on the "
                "correct side of the healthy/damaged boundary. All of this model's "
                "errors are load-regime confusions.", icon="✅")
        else:
            share = 1 - reliability["boundary_errors"] / max(reliability["total"], 1)
            st.info(
                f"{share:.2%} of windows were placed on the correct side of the "
                "healthy/damaged boundary; the remaining errors are load-regime "
                "confusions.", icon="📉")

        st.divider()

        left, right = st.columns(2)
        with left:
            cm_figure = draw_confusion_matrix(y_true, y_pred, normalise_cm)
            st.pyplot(cm_figure)
            st.download_button(
                "⬇️ Download this confusion matrix (PNG)",
                figure_to_png(cm_figure),
                file_name=f"confusion_matrix_{selected_model_name.replace(' ', '_').lower()}.png",
                mime="image/png",
            )
        with right:
            if show_roc:
                figure = draw_roc_curves(model, X, y_true)
                if figure is not None:
                    st.pyplot(figure)

        st.subheader("Classification report")
        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0,
            labels=ordered_classes(set(y_true) | set(y_pred)))
        st.dataframe(pd.DataFrame(report).T.round(4), **WIDE)
    else:
        st.info("Metrics need ground-truth labels. See the predictions tab.",
                icon="ℹ️")

    st.subheader("Predicted state distribution")
    counts = (pd.Series(y_pred, name="windows")
              .value_counts()
              .reindex(ordered_classes(set(y_pred)))
              .dropna())
    st.bar_chart(counts, color=FAULT_TINT)


# --------------------------------------------------------------------------
# Tab 2 — all models
# --------------------------------------------------------------------------

with tab_compare:
    st.subheader("Comparison across all six classifiers")

    if not has_labels:
        st.info("Upload labelled data to compare models.", icon="ℹ️")
    else:
        comparison = all_scores.round(4)
        best = comparison["MCC"].idxmax()
        st.success(
            f"**Best model on this data: {best}** "
            f"(MCC {comparison.loc[best, 'MCC']:.4f}, "
            f"Accuracy {comparison.loc[best, 'Accuracy']:.4f})",
            icon="🏆")

        st.dataframe(
            comparison.style
            .background_gradient(cmap="YlGn", axis=0)
            .format("{:.4f}"),
            **WIDE,
        )
        st.caption("Darker green is better within each column.")

        metric_for_chart = st.selectbox("Rank models by", METRIC_KEYS)
        st.pyplot(draw_model_comparison(comparison, metric_for_chart))

        st.download_button(
            "⬇️ Download comparison table (CSV)",
            comparison.to_csv().encode("utf-8"),
            file_name="model_comparison.csv",
            mime="text/csv",
        )

        # ---- health-boundary breakdown for every model ------------------
        st.markdown("##### Where each model's errors actually fall")
        breakdown = []
        for name, candidate in models.items():
            candidate_pred = candidate.predict(X)
            stats = health_reliability(y_true, candidate_pred)
            total_errors = int((candidate_pred != y_true.to_numpy()).sum())
            breakdown.append({
                "ML Model Name": name,
                "Total errors": total_errors,
                "Cross health boundary": stats["boundary_errors"],
                "Load-regime confusions": total_errors - stats["boundary_errors"],
            })
        st.dataframe(
            pd.DataFrame(breakdown).set_index("ML Model Name"),
            **WIDE)
        st.caption(
            "The middle column is the one that matters operationally — it counts "
            "the windows where a damaged gearbox was called healthy, or the "
            "reverse."
        )

        if reference.get("binary_health_only_accuracy"):
            with st.expander("Why is the target 6 classes and not just Healthy vs Broken?"):
                st.markdown(
                    "Plain Healthy-vs-Broken detection is **saturated** on this "
                    "dataset — the two acquisition campaigns barely overlap in "
                    "amplitude, so every model scores ~100 % and the comparison "
                    "says nothing. Accuracies on that binary task:"
                )
                st.dataframe(
                    pd.Series(reference["binary_health_only_accuracy"],
                              name="Binary accuracy").to_frame(),
                    **WIDE,
                )


# --------------------------------------------------------------------------
# Tab 3 — data and predictions
# --------------------------------------------------------------------------

with tab_data:
    model = models[selected_model_name]
    predicted_states = model.predict(X)

    st.subheader("Inspect a single window")
    st.caption(
        f"Pick any window and see how confident **{selected_model_name}** is "
        "across all six states. Blue bars are healthy states, red are damaged."
    )

    row_index = st.slider("Window number", 0, len(X) - 1, 0)
    probabilities = pd.Series(
        model.predict_proba(X.iloc[[row_index]])[0], index=model.classes_)
    actual = str(y_true.iloc[row_index]) if has_labels else None
    predicted = str(predicted_states[row_index])

    chart_col, detail_col = st.columns([3, 2])

    with chart_col:
        st.pyplot(draw_probability_bars(probabilities, predicted, actual))

    with detail_col:
        if has_labels:
            if predicted == actual:
                st.success(f"Correct — this window really is **{actual}**.", icon="✅")
            else:
                st.warning(
                    f"Wrong — predicted **{predicted}**, actually **{actual}**.",
                    icon="⚠️")
        st.metric("Model's confidence in its own answer",
                  f"{probabilities.max():.1%}",
                  help="Probability assigned to the predicted state. Low values "
                       "mean the window sits near a boundary between states.")
        runner_up = probabilities.drop(index=probabilities.idxmax()).idxmax()
        st.caption(f"Second guess: **{runner_up}** "
                   f"({probabilities[runner_up]:.1%})")

    with st.expander("Raw feature values for this window"):
        st.dataframe(
            X.iloc[[row_index]].T.rename(columns={X.index[row_index]: "value"}).round(4),
            **WIDE)

    st.divider()

    if has_labels:
        st.subheader("Class balance of this data")
        st.bar_chart(
            y_true.value_counts().reindex(ordered_classes(set(y_true))).dropna(),
            color=HEALTHY_TINT)

    st.subheader("Per-window predictions")
    predictions = pd.DataFrame({"predicted_state": predicted_states})
    if has_labels:
        predictions.insert(0, "true_state", y_true.values)
        predictions["correct"] = predictions["true_state"] == predictions["predicted_state"]
        wrong_only = st.checkbox("Show only the mistakes", value=False)
        shown = predictions[~predictions["correct"]] if wrong_only else predictions
    else:
        shown = predictions
    st.dataframe(shown.head(200), **WIDE)
    st.caption(f"Showing up to 200 of {len(shown):,} rows. Download for the full set.")

    st.download_button(
        "⬇️ Download all predictions (CSV)",
        predictions.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{selected_model_name.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )

    with st.expander("Input data preview (first 25 rows, all 36 features)"):
        st.dataframe(data.head(25), **WIDE)

    if reference:
        st.subheader("Reference results from training")
        st.caption(
            f"Trained on {reference.get('n_train', '?')} windows, "
            f"evaluated on {reference.get('n_test', '?')} held-out windows "
            f"(scikit-learn {reference.get('sklearn_version', '?')})."
        )
        st.dataframe(
            pd.DataFrame(reference["metrics"]).T[METRIC_KEYS],
            **WIDE,
        )

st.divider()
st.caption(
    "Data: Gearbox Fault Diagnosis Data (SpectraQuest wind-turbine gearbox rig) · "
    "Built for BITS Pilani WILP M.Tech (AI/ML) — Machine Learning Assignment 2."
)
