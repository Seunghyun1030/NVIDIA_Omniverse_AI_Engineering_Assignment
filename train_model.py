"""
Task 4 — Machine Learning Training with AutoGluon
===================================================
This script handles the complete ML training pipeline:
1. Load train/test data prepared in Task 2
2. Train AutoGluon TabularPredictor with 'best_quality' preset
3. Generate and save model leaderboard (leaderboard.csv)
4. Evaluate the best model on the hold-out test set
5. Save all model artefacts to ag_models/
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor, TabularDataset
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score,
)

# ============================================================
# Configuration
# ============================================================
TARGET = "readmitted"
MODEL_PATH = "ag_models/"
CLASS_LABELS = ["<30", ">30", "NO"]
TIME_LIMIT = 3600  # 1 hour

# ============================================================
# Step 1: Load prepared datasets
# ============================================================
print("Loading datasets...")
train_df = TabularDataset("train_data.csv")
test_df = TabularDataset("test_data.csv")

# Add sample weights to address class imbalance (inverse frequency)
class_counts = train_df[TARGET].value_counts()
total = len(train_df)
weight_map = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
train_df["sample_weight"] = train_df[TARGET].map(weight_map)
print(f"  Sample weights: {weight_map}")

print(f"  Train: {train_df.shape[0]} rows × {train_df.shape[1]} cols")
print(f"  Test:  {test_df.shape[0]} rows × {test_df.shape[1]} cols")
print(f"  Target distribution (train):\n{train_df[TARGET].value_counts()}\n")

# ============================================================
# Step 2: Train AutoGluon TabularPredictor
# ============================================================
# Using 'best_quality' preset for maximum accuracy via stacking/bagging.
# eval_metric='f1_macro' to address class imbalance (minority class <30 is 11.2%).
print("=" * 60)
print("TRAINING AutoGluon TabularPredictor")
print("=" * 60)
print(f"  Preset:      best_quality")
print(f"  Eval metric: f1_macro")
print(f"  Time limit:  {TIME_LIMIT}s ({TIME_LIMIT // 60} min)")
print(f"  Save path:   {MODEL_PATH}")
print()

predictor = TabularPredictor(
    label=TARGET,
    eval_metric="f1_macro",
    path=MODEL_PATH,
    verbosity=2,
    sample_weight="sample_weight",
    weight_evaluation=False,
).fit(
    train_data=train_df,
    presets="best_quality",
    time_limit=TIME_LIMIT,
)

print("\nTraining complete.\n")

# ============================================================
# Step 3: Generate and save model leaderboard
# ============================================================
print("=" * 60)
print("MODEL LEADERBOARD")
print("=" * 60)

leaderboard = predictor.leaderboard(
    data=test_df,
    extra_info=True,
    extra_metrics=["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "roc_auc_ovo_macro"],
)
leaderboard.to_csv("leaderboard.csv", index=False)
print(f"\nLeaderboard saved to leaderboard.csv ({len(leaderboard)} models)")
display_cols = ["model", "score_test", "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "roc_auc_ovo_macro", "fit_time"]
print(leaderboard[[c for c in display_cols if c in leaderboard.columns]].to_string())
print()

# ============================================================
# Step 4: Evaluate best model on hold-out test set
# ============================================================
print("=" * 60)
print("TEST SET EVALUATION (Best Model)")
print("=" * 60)

# AutoGluon's built-in evaluation
eval_results = predictor.evaluate(data=test_df, auxiliary_metrics=True)
print(f"\nAutoGluon evaluation results:")
for metric, score in eval_results.items():
    print(f"  {metric}: {score:.4f}")

# Detailed sklearn-based evaluation
y_true = test_df[TARGET]
y_pred = predictor.predict(test_df)

# Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"\nAccuracy: {acc:.4f}")

# F1 Score (Macro)
f1_mac = f1_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0)
print(f"F1 Score (Macro): {f1_mac:.4f}")

# Classification Report (per-class precision, recall, F1)
report_text = classification_report(
    y_true, y_pred,
    labels=CLASS_LABELS,
    digits=4,
    zero_division=0,
)
print(f"\nClassification Report:\n{report_text}")

# Save classification report as CSV
report_dict = classification_report(
    y_true, y_pred,
    labels=CLASS_LABELS,
    output_dict=True,
    zero_division=0,
)
report_df = pd.DataFrame(report_dict).transpose()
os.makedirs("metrics", exist_ok=True)
os.makedirs("charts", exist_ok=True)
report_df.to_csv("metrics/classification_report.csv")

# Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ConfusionMatrixDisplay.from_predictions(
    y_true, y_pred,
    labels=CLASS_LABELS,
    display_labels=CLASS_LABELS,
    cmap="Blues",
    values_format="d",
    ax=axes[0],
)
axes[0].set_title("Confusion Matrix (Counts)", fontsize=13)

ConfusionMatrixDisplay.from_predictions(
    y_true, y_pred,
    labels=CLASS_LABELS,
    display_labels=CLASS_LABELS,
    normalize="true",
    cmap="Blues",
    values_format=".2%",
    ax=axes[1],
)
axes[1].set_title("Confusion Matrix (Normalized by True Label)", fontsize=13)

plt.tight_layout()
plt.savefig("charts/confusion_matrix.png", dpi=150, bbox_inches="tight")
print("Confusion matrix saved to charts/confusion_matrix.png")

# Save all metrics to JSON
metrics = {
    "accuracy": float(acc),
    "f1_macro": float(f1_mac),
    "per_class": {
        label: {
            "precision": report_dict[label]["precision"],
            "recall": report_dict[label]["recall"],
            "f1_score": report_dict[label]["f1-score"],
            "support": int(report_dict[label]["support"]),
        }
        for label in CLASS_LABELS
    },
}

with open("metrics/evaluation_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Metrics saved to metrics/evaluation_metrics.json")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("TASK 4 SUMMARY")
print("=" * 60)
print(f"Best model:       {predictor.model_best}")
print(f"Accuracy:         {acc:.4f}")
print(f"F1 Macro:         {f1_mac:.4f}")
print(f"Models trained:   {len(leaderboard)}")
print(f"Leaderboard:      leaderboard.csv")
print(f"Model artefacts:  {MODEL_PATH}")
print(f"Confusion matrix: charts/confusion_matrix.png")
print(f"Metrics:          metrics/evaluation_metrics.json")
print("=" * 60)
