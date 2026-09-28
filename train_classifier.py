# =============================================================================
# train_classifier.py
# Trains and compares a Random Forest and SVM classifier for activity
# recognition, then saves the best-performing model alongside its fitted
# scaler and label encoder.
#
# Usage:
#   python train_classifier.py
#   python train_classifier.py --data path/to/custom.csv
#
# Outputs (written to models/ and data/):
#   models/classifier.pkl      — best model (RandomForest or SVM)
#   models/scaler.pkl          — fitted StandardScaler
#   models/label_encoder.pkl   — fitted LabelEncoder (int ↔ class name)
#   data/confusion_matrix.png  — Seaborn heatmap for the best model
# =============================================================================

import argparse
import logging
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on headless machines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV     = os.path.join(BASE_DIR, "data", "training_data.csv")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
MODEL_PATH      = os.path.join(MODELS_DIR, "classifier.pkl")
SCALER_PATH     = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODER_PATH    = os.path.join(MODELS_DIR, "label_encoder.pkl")
CM_PATH         = os.path.join(BASE_DIR, "data", "confusion_matrix.png")

# Minimum samples per class before a warning is shown.
MIN_SAMPLES_WARN = 30

# ---------------------------------------------------------------------------
# Canonical feature columns — must exactly match modules/feature_extractor.py
# output + the motion_variance field computed in extract_ucf_features.py.
# Training will use ONLY these columns regardless of what else is in the CSV.
# zone_dwell_time (legacy synthetic column) is intentionally excluded.
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "avg_speed",
    "max_speed",
    "total_displacement",
    "total_distance",
    "stillness_ratio",
    "pace_ratio",
    "motion_variance",
]


# ===========================================================================
# Data loading
# ===========================================================================

def load_and_clean(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the CSV, drop inf / NaN values, and return (X, y).

    Only the canonical FEATURE_COLS are used as input features — any extra
    columns present in the CSV (e.g. legacy zone_dwell_time) are ignored.
    Rows that are missing any required column value are dropped.
    """
    logger.info("Loading dataset from '%s'...", csv_path)

    if not os.path.isfile(csv_path):
        logger.error("CSV not found: '%s'", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        logger.error("No 'label' column found in '%s'.", csv_path)
        sys.exit(1)

    # Warn about any canonical feature columns that are missing.
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        logger.error(
            "CSV is missing required feature columns: %s  |  Present: %s",
            missing_cols, list(df.columns),
        )
        sys.exit(1)

    # Note any extra / legacy columns being ignored.
    extra_cols = [c for c in df.columns if c != "label" and c not in FEATURE_COLS]
    if extra_cols:
        logger.info("Ignoring %d legacy column(s) not used for training: %s", len(extra_cols), extra_cols)

    # -- Replace inf with NaN then drop rows missing required values ----------
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(subset=FEATURE_COLS + ["label"], inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows containing NaN / Inf values.", dropped)

    X = df[FEATURE_COLS]
    y = df["label"]

    logger.info("Dataset ready -- %d samples, %d features.", len(df), len(FEATURE_COLS))
    logger.info("Feature columns: %s", FEATURE_COLS)
    return X, y


# ===========================================================================
# Class distribution report
# ===========================================================================

def report_class_distribution(y: pd.Series) -> None:
    """Print class counts and warn if any class is under-represented."""
    counts = y.value_counts().sort_index()
    total  = len(y)

    print("\n" + "-" * 50)
    print("  Class Distribution")
    print("-" * 50)
    max_label_len = max(len(str(lbl)) for lbl in counts.index)
    for label, count in counts.items():
        bar   = "#" * min(40, count // max(1, total // 40))
        pct   = count / total * 100
        print(f"  {str(label):<{max_label_len}}  {count:>5}  ({pct:5.1f}%)  {bar}")
    print("-" * 50)
    print(f"  Total samples : {total}")
    print("-" * 50)

    # -- Imbalance warning -----------------------------------------------------
    low = [lbl for lbl, cnt in counts.items() if cnt < MIN_SAMPLES_WARN]
    if low:
        print(
            f"\n  [!]  WARNING: The following classes have fewer than "
            f"{MIN_SAMPLES_WARN} samples and may produce unreliable results:"
        )
        for lbl in low:
            print(f"       • {lbl}  ({counts[lbl]} sample(s))")
        print(
            "     Consider collecting more data or using --oversample.\n"
        )


# ===========================================================================
# Training helpers
# ===========================================================================

def train_and_evaluate(
    name: str,
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_enc: np.ndarray,
    y_test_enc: np.ndarray,
    class_names: list[str],
) -> tuple[float, np.ndarray]:
    """
    Fit *model* on the training set, evaluate on the test set, and print
    the accuracy + full classification report.

    Returns (accuracy, y_pred_encoded).
    """
    print(f"\n{'=' * 55}")
    print(f"  Model : {name}")
    print(f"{'=' * 55}")

    logger.info("Fitting %s…", name)
    model.fit(X_train, y_train_enc)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test_enc, y_pred)

    print(f"  Overall Accuracy : {acc * 100:.2f}%\n")
    print("  Classification Report:")
    print(
        classification_report(
            y_test_enc, y_pred,
            target_names=class_names,
            zero_division=0,
        )
    )

    return acc, y_pred


# ===========================================================================
# Confusion matrix plot
# ===========================================================================

def save_confusion_matrix(
    y_true_enc: np.ndarray,
    y_pred_enc: np.ndarray,
    class_names: list[str],
    model_name: str,
    out_path: str,
) -> None:
    """Save a Seaborn heatmap of the confusion matrix with real label strings."""
    cm = confusion_matrix(y_true_enc, y_pred_enc)

    fig, ax = plt.subplots(figsize=(max(7, len(class_names) * 1.8), max(6, len(class_names) * 1.5)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label",      fontsize=12, labelpad=10)
    ax.set_title(
        f"Confusion Matrix — {model_name}",
        fontsize=14, pad=14,
    )
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Confusion matrix saved to '%s'.", out_path)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train activity classifier (Random Forest vs SVM) and save the best model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_classifier.py
  python train_classifier.py --data data/my_labels.csv
        """,
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_CSV,
        help=f"Path to the labelled CSV (default: {DEFAULT_CSV}).",
    )
    args = parser.parse_args()

    # -- Load & clean ----------------------------------------------------------
    X, y = load_and_clean(args.data)

    # -- Class distribution ----------------------------------------------------
    report_class_distribution(y)

    # -- Encode string labels -> integers ---------------------------------------
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)   # e.g. ["Loitering", "Normal", "Running", …]

    print("\n  Label encoding map:")
    for i, name in enumerate(class_names):
        print(f"    {i}  ->  {name}")

    # -- 80/20 stratified split ------------------------------------------------
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=0.20,
        stratify=y_enc,
        random_state=42,
    )
    logger.info(
        "Train/test split — train: %d, test: %d samples.",
        len(X_train_df), len(X_test_df),
    )

    # -- Scale features (fit on train only) ------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test  = scaler.transform(X_test_df)

    # -- Define models ---------------------------------------------------------
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    svm_model = SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42,
    )

    # -- Train & evaluate both models ------------------------------------------
    rf_acc,  rf_pred  = train_and_evaluate(
        "Random Forest", rf_model,
        X_train, X_test, y_train, y_test, class_names,
    )
    svm_acc, svm_pred = train_and_evaluate(
        "SVM (RBF kernel)", svm_model,
        X_train, X_test, y_train, y_test, class_names,
    )

    # -- Pick the better model -------------------------------------------------
    if rf_acc >= svm_acc:
        best_name  = "Random Forest"
        best_model = rf_model
        best_pred  = rf_pred
        best_acc   = rf_acc
    else:
        best_name  = "SVM"
        best_model = svm_model
        best_pred  = svm_pred
        best_acc   = svm_acc

    print(f"\n  >>  Best model: {best_name}  ({best_acc * 100:.2f}%)")

    # -- Confusion matrix for the best model -----------------------------------
    save_confusion_matrix(
        y_test, best_pred,
        class_names=class_names,
        model_name=best_name,
        out_path=CM_PATH,
    )

    # -- Persist artefacts -----------------------------------------------------
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(best_model, MODEL_PATH,   compress=3)
    joblib.dump(scaler,     SCALER_PATH,  compress=3)
    joblib.dump(le,         ENCODER_PATH, compress=3)

    logger.info("classifier  -> %s  (%.1f KB)", MODEL_PATH,   os.path.getsize(MODEL_PATH)   / 1024)
    logger.info("scaler      -> %s  (%.1f KB)", SCALER_PATH,  os.path.getsize(SCALER_PATH)  / 1024)
    logger.info("label enc.  -> %s  (%.1f KB)", ENCODER_PATH, os.path.getsize(ENCODER_PATH) / 1024)

    # -- Final summary ---------------------------------------------------------
    print("\n" + "=" * 55)
    print(f"  [OK] Best model        : {best_name}")
    print(f"  [OK] Test accuracy     : {best_acc * 100:.1f}%")
    print(f"  [OK] Model saved to    : {os.path.relpath(MODEL_PATH)}")
    print(f"  [OK] Scaler saved to   : {os.path.relpath(SCALER_PATH)}")
    print(f"  [OK] Label encoder     : {os.path.relpath(ENCODER_PATH)}")
    print(f"  [OK] Confusion matrix  : {os.path.relpath(CM_PATH)}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
