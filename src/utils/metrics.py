"""
Evaluation and visualization utilities for sequence labeling.
Computes seqeval Span-Level F1/Precision/Recall, plots CRF transition heatmaps,
and generates token-level Confusion Matrices.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from seqeval.metrics import (
    classification_report as seqeval_report,
    f1_score as seqeval_f1,
    precision_score as seqeval_precision,
    recall_score as seqeval_recall,
)
from seqeval.scheme import IOB2

FIGURES_DIR = Path("reports/figures")

CORE_BIO_TAGS = [
    "O",
    "B-INSULT", "I-INSULT",
    "B-IDENTITY_ATTACK", "I-IDENTITY_ATTACK",
    "B-OBSCENE", "I-OBSCENE",
    "B-THREAT", "I-THREAT",
    "B-OTHER_TOXIC", "I-OTHER_TOXIC"
]

def evaluate_predictions(y_true: List[List[str]], y_pred: List[List[str]]) -> Dict[str, Any]:
    """
    Computes strict span-level metrics using seqeval (IOB2 scheme) without warnings.
    """
    f1 = seqeval_f1(y_true, y_pred, mode="strict", scheme=IOB2, zero_division=0)
    prec = seqeval_precision(y_true, y_pred, mode="strict", scheme=IOB2, zero_division=0)
    rec = seqeval_recall(y_true, y_pred, mode="strict", scheme=IOB2, zero_division=0)
    report = seqeval_report(y_true, y_pred, mode="strict", scheme=IOB2, digits=4, zero_division=0)
    
    # Token-level accuracy
    total_tokens = sum(len(t) for t in y_true)
    correct_tokens = sum(
        sum(1 for yt, yp in zip(t_seq, p_seq) if yt == yp)
        for t_seq, p_seq in zip(y_true, y_pred)
    )
    token_acc = (correct_tokens / total_tokens) if total_tokens > 0 else 0.0

    return {
        "strict_span_f1": f1,
        "strict_span_precision": prec,
        "strict_span_recall": rec,
        "token_accuracy": token_acc,
        "classification_report": report
    }

def plot_transition_matrix(
    matrix: np.ndarray, 
    labels: List[str], 
    title: str = "Learned Transition Matrix",
    filename: str = "transition_matrix.png",
    clip_range: Tuple[float, float] = (-6.0, 4.0)
):
    """
    Plots and saves a clean heatmap of transition probabilities/scores.
    Filters out special tokens (<START>, <STOP>, <PAD>) to prevent extreme scale skewing.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / filename
    
    # Identify non-special tokens
    valid_indices = [i for i, label in enumerate(labels) if not label.startswith("<")]
    clean_labels = [labels[i] for i in valid_indices]
    clean_matrix = matrix[np.ix_(valid_indices, valid_indices)].copy()
    
    # Clip extreme values for visual clarity if values are logits/unbounded
    if clean_matrix.min() < -10.0 or clean_matrix.max() > 10.0:
        clean_matrix = np.clip(clean_matrix, clip_range[0], clip_range[1])

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        clean_matrix, 
        xticklabels=clean_labels, 
        yticklabels=clean_labels, 
        cmap="coolwarm", 
        annot=True, 
        fmt=".2f",
        cbar=True,
        linewidths=0.5
    )
    plt.title(title, fontsize=13, pad=12, fontweight="bold")
    plt.xlabel("To Tag (y_t)", fontsize=11)
    plt.ylabel("From Tag (y_{t-1})", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Transition matrix heatmap saved to: {out_path}")
    return out_path

def plot_confusion_matrix(
    y_true: List[List[str]], 
    y_pred: List[List[str]], 
    title: str = "Token-Level Confusion Matrix",
    filename: str = "confusion_matrix.png",
    normalize: bool = True
):
    """
    Flattens token tags, computes scikit-learn confusion matrix across the 11 classes,
    and saves an annotated heatmap.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / filename
    
    # Flatten sequences
    flat_true = [tag for seq in y_true for tag in seq]
    flat_pred = [tag for seq in y_pred for tag in seq]
    
    # Use standard 11 classes present in dataset
    all_classes = [c for c in CORE_BIO_TAGS if c in set(flat_true).union(set(flat_pred))]
    
    cm = confusion_matrix(flat_true, flat_pred, labels=all_classes)
    
    if normalize:
        # Normalize by row (true class count) to show recall per class
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_to_plot = cm.astype("float") / row_sums
        fmt = ".2f"
        cbar_label = "Proportion (Recall)"
    else:
        cm_to_plot = cm
        fmt = "d"
        cbar_label = "Token Count"
        
    plt.figure(figsize=(11, 9))
    sns.heatmap(
        cm_to_plot,
        xticklabels=all_classes,
        yticklabels=all_classes,
        cmap="Blues",
        annot=True,
        fmt=fmt,
        cbar_kws={"label": cbar_label},
        linewidths=0.5
    )
    plt.title(f"{title} ({'Normalized' if normalize else 'Counts'})", fontsize=13, pad=12, fontweight="bold")
    plt.xlabel("Predicted Tag", fontsize=11)
    plt.ylabel("True Gold Tag", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Confusion matrix heatmap saved to: {out_path}")
    return out_path
