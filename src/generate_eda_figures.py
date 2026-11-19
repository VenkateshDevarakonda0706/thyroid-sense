"""Exploratory Data Analysis (EDA) Script for ThyroidSense (Thyroid Disease Detection).

Generates and saves publication-quality exploratory visualizations to results/figures/:
1. class_distribution.png: Target diagnosis frequencies.
2. feature_distributions.png: Histograms and KDE curves for continuous lab features.
3. correlation_heatmap.png: Annotated Pearson correlation matrix.
4. hormone_boxplots_by_class.png: TSH, T3, TT4 distributions grouped by thyroid condition.
5. raw_missing_values_audit.png: Missing value percentages in the raw clinical records.
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set visual styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

FIGURES_DIR = os.path.join("results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

PROCESSED_DATA_PATH = os.path.join("data", "processed", "thyroid_processed.csv")
RAW_DATA_PATH = os.path.join("data", "raw", "thyroid_raw.csv")

CLASS_NAMES = {
    1: "Compensated\nHypothyroid",
    7: "Negative\n(Normal)",
    9: "Primary\nHypothyroid",
    11: "Secondary\nHypothyroid",
}


def plot_class_distribution(df: pd.DataFrame) -> str:
    """Plot target class distribution in the modeling dataset."""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df["class"].value_counts().sort_index()
    labels = [CLASS_NAMES.get(k, str(k)) for k in counts.index]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    bars = ax.bar(labels, counts.values, color=colors, edgecolor="black", linewidth=1.2)
    ax.set_title("Target Class Distribution (Curated Dataset)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Thyroid Diagnostic Category", fontsize=12, labelpad=8)
    ax.set_ylabel("Number of Patient Records", fontsize=12, labelpad=8)
    ax.set_ylim(0, max(counts.values) * 1.15)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    out_path = os.path.join(FIGURES_DIR, "class_distribution.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_feature_distributions(df: pd.DataFrame) -> str:
    """Plot histograms and KDE distributions for numerical clinical tests."""
    num_cols = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, col in enumerate(num_cols):
        ax = axes[idx]
        sns.histplot(df[col], kde=True, ax=ax, color="#2b5c8f", bins=30, edgecolor="white")
        skewness = df[col].skew()
        ax.set_title(f"Distribution of {col} (Skew: {skewness:.2f})", fontsize=12, fontweight="bold")
        ax.set_xlabel(f"{col} Measurement", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)

    fig.suptitle("Numerical Clinical and Hormone Feature Distributions", fontsize=16, fontweight="bold", y=1.02)
    out_path = os.path.join(FIGURES_DIR, "feature_distributions.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    """Plot Pearson correlation matrix for all features and target."""
    fig, ax = plt.subplots(figsize=(13, 10))
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
        ax=ax,
    )
    ax.set_title("Pearson Correlation Heatmap of Thyroid Features", fontsize=15, fontweight="bold", pad=12)
    out_path = os.path.join(FIGURES_DIR, "correlation_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_hormone_boxplots_by_class(df: pd.DataFrame) -> str:
    """Plot comparative boxplots showing hormone levels across diagnostic classes."""
    plot_df = df.copy()
    plot_df["diagnosis"] = plot_df["class"].map(CLASS_NAMES)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    hormones = [("TSH", "Thyroid-Stimulating Hormone (mIU/L)"),
                ("T3", "Triiodothyronine (nmol/L)"),
                ("TT4", "Total Thyroxine (nmol/L)"),
                ("FTI", "Free Thyroxine Index")]

    for idx, (col, label) in enumerate(hormones):
        ax = axes[idx // 2, idx % 2]
        sns.boxplot(
            data=plot_df,
            x="diagnosis",
            y=col,
            ax=ax,
            palette="Set2",
            showfliers=False,  # Suppress extreme outliers for readable clinical range view
        )
        ax.set_title(f"{col} Levels by Diagnosis", fontsize=12, fontweight="bold")
        ax.set_xlabel("Clinical Diagnosis", fontsize=10)
        ax.set_ylabel(label, fontsize=10)

    fig.suptitle("Thyroid Hormone Variation Across Diagnostic Classes", fontsize=16, fontweight="bold", y=1.02)
    out_path = os.path.join(FIGURES_DIR, "hormone_boxplots_by_class.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_raw_missing_values(raw_path: str = RAW_DATA_PATH) -> str:
    """Plot missing value percentages in the raw dataset."""
    if not os.path.exists(raw_path):
        return ""
    raw_df = pd.read_csv(raw_path)
    raw_df.replace("?", np.nan, inplace=True)
    missing_pct = (raw_df.isnull().sum() / len(raw_df)) * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)

    if missing_pct.empty:
        return ""

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(missing_pct.index, missing_pct.values, color="#d95f02", edgecolor="black")
    ax.set_title("Missing Value Percentages in Raw UCI Thyroid Data", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Percentage Missing (%)", fontsize=12)
    ax.set_ylabel("Dataset Columns", fontsize=12)
    ax.set_xlim(0, max(missing_pct.values) * 1.15)

    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f"{width:.1f}%",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=10,
        )

    out_path = os.path.join(FIGURES_DIR, "raw_missing_values_audit.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def run_all_eda():
    """Execute all EDA visualizations."""
    print("Loading processed dataset for EDA...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    plot_class_distribution(df)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_hormone_boxplots_by_class(df)
    plot_raw_missing_values(RAW_DATA_PATH)
    print("All exploratory figures generated successfully in results/figures/")


if __name__ == "__main__":
    run_all_eda()

