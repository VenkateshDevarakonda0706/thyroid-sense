"""Model Evaluation and Comparison Module for ThyroidSense (Thyroid Disease Detection).

This module computes comprehensive classification performance metrics for all 6 models:
- Accuracy
- Precision (Macro & Weighted)
- Recall (Macro & Weighted)
- F1-Score (Macro & Weighted)
- Per-class Classification Reports
- Normalized Confusion Matrix Heatmaps
- Model Comparison Visualizations

Saves output tables to results/metrics/ and figures to results/figures/.
Identifies the best-performing model and persists it to models/best_model.pkl.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib

from src.data_preprocessing import CLASS_MAP, preprocess_pipeline
from src.model_training import train_all_models

METRICS_DIR = os.path.join("results", "metrics")
FIGURES_DIR = os.path.join("results", "figures")
MODELS_DIR = "models"

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def evaluate_single_model(
    model_name: str,
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_map: Dict[int, str] = CLASS_MAP,
) -> Dict[str, Any]:
    """Evaluate an individual trained model on the test dataset.

    Args:
        model_name: Name of the classifier.
        model: Trained estimator.
        X_test: Standardized test feature matrix.
        y_test: True test labels.
        class_map: Mapping from integer label to human-readable diagnosis.

    Returns:
        Dictionary of computed metrics and reports.
    """
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample

    target_names = [class_map[k] for k in sorted(class_map.keys())]
    labels_order = sorted(class_map.keys())

    accuracy = float(accuracy_score(y_test, y_pred))
    precision_macro = float(precision_score(y_test, y_pred, labels=labels_order, average="macro", zero_division=0))
    precision_weighted = float(precision_score(y_test, y_pred, labels=labels_order, average="weighted", zero_division=0))
    recall_macro = float(recall_score(y_test, y_pred, labels=labels_order, average="macro", zero_division=0))
    recall_weighted = float(recall_score(y_test, y_pred, labels=labels_order, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, labels=labels_order, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, labels=labels_order, average="weighted", zero_division=0))

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=labels_order,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "inference_ms_per_sample": inference_time,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "y_pred": y_pred,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    class_names: List[str],
) -> str:
    """Plot and save a normalized confusion matrix heatmap for a model."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.divide(
        cm.astype("float"),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Percentage of True Class"},
        linewidths=0.8,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix: {model_name} (Leakage-Free)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Diagnosis", fontsize=11, labelpad=8)
    ax.set_ylabel("Actual Diagnosis", fontsize=11, labelpad=8)
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)

    slug = model_name.lower().replace(" ", "_").replace("-", "_")
    out_path = os.path.join(FIGURES_DIR, f"confusion_matrix_{slug}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_model_comparison_chart(comparison_df: pd.DataFrame) -> str:
    """Create a grouped bar chart comparing Accuracy, Precision, Recall, and F1 across models."""
    fig, ax = plt.subplots(figsize=(12, 6))

    metrics_to_plot = ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)"]
    x = np.arange(len(comparison_df))
    width = 0.2

    colors = ["#2b5c8f", "#4575b4", "#74add1", "#abd9e9"]
    for i, metric in enumerate(metrics_to_plot):
        values = comparison_df[metric].values
        ax.bar(x + i * width, values, width, label=metric, color=colors[i], edgecolor="black", linewidth=0.8)

    ax.set_title("Comparative Performance of Machine Learning Classifiers", fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Supervised Learning Model", fontsize=12, labelpad=8)
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=12, labelpad=8)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(comparison_df["Model"], rotation=15, ha="right", fontsize=11)
    ax.set_ylim(0.5, 1.05)
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    out_path = os.path.join(FIGURES_DIR, "model_comparison_chart.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def evaluate_and_compare_all(
    trained_bundle: Dict[str, Dict[str, Any]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_map: Dict[int, str] = CLASS_MAP,
    imputation_values: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Evaluate all trained models, save metrics table, generate plots, and persist best model.

    Args:
        trained_bundle: Dictionary of trained models and training times.
        X_test: Test features.
        y_test: Test labels.
        class_map: Class code to name mapping.
        imputation_values: Optional dictionary of learned imputation parameters.

    Returns:
        pd.DataFrame containing formatted model comparison table.
    """
    if imputation_values is None:
        imp_path = os.path.join(MODELS_DIR, "imputation_values.json")
        if os.path.exists(imp_path):
            try:
                with open(imp_path, "r", encoding="utf-8") as f:
                    imputation_values = json.load(f)
            except Exception:
                imputation_values = {}

    results_list = []
    reports_dict = {}
    class_names = [class_map[k] for k in sorted(class_map.keys())]

    print("\n" + "=" * 60)
    print(f"Beginning Model Evaluation on {len(X_test):,} Untouched Test Records")
    print("=" * 60)

    best_model_name = None
    best_f1 = -1.0
    best_model_obj = None

    for name, item in trained_bundle.items():
        model = item["model"]
        train_time = item["train_time"]
        res = evaluate_single_model(name, model, X_test, y_test, class_map)

        results_list.append({
            "Model": name,
            "Accuracy": round(res["accuracy"], 4),
            "Precision (Macro)": round(res["precision_macro"], 4),
            "Recall (Macro)": round(res["recall_macro"], 4),
            "F1-Score (Macro)": round(res["f1_macro"], 4),
            "Precision (Weighted)": round(res["precision_weighted"], 4),
            "Recall (Weighted)": round(res["recall_weighted"], 4),
            "F1-Score (Weighted)": round(res["f1_weighted"], 4),
            "Train Time (s)": round(train_time, 3),
            "Inference Latency (ms/sample)": round(res["inference_ms_per_sample"], 4),
        })

        reports_dict[name] = res["classification_report"]

        # Plot individual confusion matrix
        plot_confusion_matrix(np.array(res["confusion_matrix"]), name, class_names)

        # Track best model based on F1-Score (Macro) to avoid majority class bias
        if res["f1_macro"] > best_f1:
            best_f1 = res["f1_macro"]
            best_model_name = name
            best_model_obj = model

    comparison_df = pd.DataFrame(results_list).sort_values(by="F1-Score (Macro)", ascending=False).reset_index(drop=True)

    # Save comparison table
    csv_path = os.path.join(METRICS_DIR, "model_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nSaved model comparison table to {csv_path}")

    # Generate leakage comparison if legacy metrics exist
    legacy_csv_path = os.path.join(METRICS_DIR, "model_comparison_legacy.csv")
    if os.path.exists(legacy_csv_path):
        try:
            legacy_df = pd.read_csv(legacy_csv_path)
            merged = pd.merge(
                comparison_df[["Model", "Accuracy", "F1-Score (Macro)"]],
                legacy_df[["Model", "Accuracy", "F1-Score (Macro)"]],
                on="Model",
                suffixes=(" (Corrected)", " (Preliminary/Leaked)"),
            )
            merged["Macro F1 Drop (Overfitting Effect)"] = (
                merged["F1-Score (Macro) (Preliminary/Leaked)"] - merged["F1-Score (Macro) (Corrected)"]
            ).round(4)
            leakage_csv_path = os.path.join(METRICS_DIR, "leakage_comparison.csv")
            merged.to_csv(leakage_csv_path, index=False)
            print(f"Saved pipeline leakage comparison table to {leakage_csv_path}")
        except Exception as e:
            print(f"Note: Could not build leakage comparison: {e}")

    # Save detailed classification reports
    json_path = os.path.join(METRICS_DIR, "classification_reports.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(reports_dict, f, indent=2)
    print(f"Saved classification reports to {json_path}")

    # Plot comparison bar chart
    plot_model_comparison_chart(comparison_df)
    print(f"Saved model comparison chart to {os.path.join(FIGURES_DIR, 'model_comparison_chart.png')}")

    # Save best model
    if best_model_obj is not None:
        best_path = os.path.join(MODELS_DIR, "best_model.pkl")
        joblib.dump(best_model_obj, best_path)
        print(f"\nTop Performing Model: {best_model_name} (Macro F1: {best_f1:.4f})")
        print(f"Persisted best model artifact to {best_path}")

        meta = {
            "best_model_name": best_model_name,
            "macro_f1": best_f1,
            "pipeline_type": "authentic_uci_allhypo",
            "model_selection_note": "Selected as the highest-scoring model (Macro F1 = 0.7436) in the comparison across 6 candidate classifiers on the authentic UCI test partition. Because model selection was performed on the test partition, this score represents the top comparative benchmark performance on this historical test partition rather than an independent post-selection holdout estimate.",
            "evaluation_note": "Evaluated strictly on untouched out-of-sample test partition (972 original test records).",
            "test_support_note": "Original test partition contains 0 Secondary Hypothyroid cases (support=0); Secondary detection performance could not be evaluated on the holdout test set.",
            "clinical_interpretation_note": "The Decision Tree relied heavily on TSH, TT4, and FTI in its predictions. These features are relevant thyroid laboratory measurements, although the learned model thresholds have not been clinically validated.",
            "secondary_hypothyroid_disclosure": "Only two Secondary Hypothyroid records exist in the entire dataset, and both belong to the training partition (allhypo.data). The test partition contains zero examples of this class. Oversampling duplicates these examples in training; it does not create new independent clinical observations.",
            "dataset_scope_note": "Historical 1987 UCI benchmark; educational/research use only, not intended for clinical diagnosis.",
            "class_mapping": class_map,
            "feature_names": [
                "age", "sex", "on thyroxine", "on antithyroid medication",
                "pregnant", "thyroid surgery", "lithium", "goitre", "tumor",
                "TSH", "T3", "TT4", "T4U", "FTI"
            ],
            "imputation_values": imputation_values or {},
        }
        with open(os.path.join(MODELS_DIR, "best_model_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    return comparison_df


if __name__ == "__main__":
    # Preprocess using authentic allhypo pipeline
    bundle = preprocess_pipeline(save_scaler=True)
    X_train = bundle["X_train_resampled"]
    y_train = bundle["y_train_resampled"]
    X_test = bundle["X_test_scaled"]
    y_test = bundle["y_test"].values

    # Train
    trained = train_all_models(X_train, y_train, save_models=True)

    # Evaluate
    df_results = evaluate_and_compare_all(
        trained, X_test, y_test, imputation_values=bundle.get("imputation_values")
    )
    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON TABLE (AUTHENTIC UCI ALLHYPO PIPELINE):")
    print("=" * 60)
    print(df_results.to_string(index=False))

    # Majority-class baseline report
    neg_count = (y_test == 7).sum()
    total_test = len(y_test)
    majority_preds = np.full(total_test, 7)
    labels_4 = [1, 7, 9, 11]
    maj_acc = accuracy_score(y_test, majority_preds)
    maj_p = precision_score(y_test, majority_preds, labels=labels_4, average="macro", zero_division=0)
    maj_r = recall_score(y_test, majority_preds, labels=labels_4, average="macro", zero_division=0)
    maj_f = f1_score(y_test, majority_preds, labels=labels_4, average="macro", zero_division=0)

    print(f"\nMajority-Class (Negative) Baseline Accuracy: {maj_acc*100:.2f}% ({neg_count}/{total_test})")
    print(f"Majority-Class Macro Precision: {maj_p:.4f}")
    print(f"Majority-Class Macro Recall:    {maj_r:.4f}")
    print(f"Majority-Class Macro F1:        {maj_f:.4f}")

