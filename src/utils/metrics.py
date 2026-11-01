"""
Evaluation and visualization utilities for sequence labeling.
Computes seqeval Span-Level F1/Precision/Recall, plots CRF transition heatmaps,
and generates both Full and Toxic-Only token-level Confusion Matrices.
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

# Order with toxic categories first and 'O' at the END to keep visual diagonal intuitive
ORDERED_BIO_TAGS = [
    "B-INSULT", "I-INSULT",
    "B-IDENTITY_ATTACK", "I-IDENTITY_ATTACK",
    "B-OBSCENE", "I-OBSCENE",
    "B-THREAT", "I-THREAT",
    "B-OTHER_TOXIC", "I-OTHER_TOXIC",
    "O"
]

TOXIC_ONLY_TAGS = [
    "B-INSULT", "I-INSULT",
    "B-IDENTITY_ATTACK", "I-IDENTITY_ATTACK",
    "B-OBSCENE", "I-OBSCENE",
    "B-THREAT", "I-THREAT",
    "B-OTHER_TOXIC", "I-OTHER_TOXIC"
]

def evaluate_predictions(y_true: List[List[str]], y_pred: List[List[str]]) -> Dict[str, Any]:
    """Computes strict span-level metrics using seqeval (IOB2 scheme)."""
    f1 = seqeval_f1(y_true, y_pred, mode="strict", scheme=IOB2, zero_division=0)
    prec = seqeval_precision(y_true, y_pred, mode="strict", scheme=IOB2, zero_division=0)
    rec = seqeval_recall(y_true, y_pred, mode="strict", scheme=IOB2, zero_division=0)
    report = seqeval_report(y_true, y_pred, mode="strict", scheme=IOB2, digits=4, zero_division=0)
    
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
    Filters out special tokens (<START>, <STOP>, <PAD>).
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / filename
    
    valid_indices = [i for i, label in enumerate(labels) if not label.startswith("<")]
    clean_labels = [labels[i] for i in valid_indices]
    clean_matrix = matrix[np.ix_(valid_indices, valid_indices)].copy()
    
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
    return out_path

def plot_confusion_matrix(
    y_true: List[List[str]], 
    y_pred: List[List[str]], 
    title: str = "Token-Level Confusion Matrix",
    filename: str = "confusion_matrix.png",
    normalize: bool = True,
    toxic_only: bool = False
):
    """
    Flattens token tags and computes confusion matrix.
    Computes full 11-class matrix first so row sums reflect ALL true gold tokens.
    toxic_only: If True, zooms into the 10 toxic classes, displaying true class recall
                on the diagonal and true cross-class confusion rates, avoiding discrete artifacts.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / filename
    
    flat_true = [tag for seq in y_true for tag in seq]
    flat_pred = [tag for seq in y_pred for tag in seq]
    
    # Always compute over all 11 classes first to capture total true support per class
    cm_full = confusion_matrix(flat_true, flat_pred, labels=ORDERED_BIO_TAGS)
    
    if normalize:
        row_sums = cm_full.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm_full.astype("float") / row_sums
    else:
        cm_norm = cm_full
        
    if toxic_only:
        # Keep the 10 toxic classes (first 10 of ORDERED_BIO_TAGS)
        active_classes = TOXIC_ONLY_TAGS
        cm_to_plot = cm_norm[:10, :10]
        fmt = ".2f" if normalize else "d"
        cbar_label = "Recall per Class" if normalize else "Token Count"
    else:
        active_classes = ORDERED_BIO_TAGS
        cm_to_plot = cm_norm
        fmt = ".2f" if normalize else "d"
        cbar_label = "Recall per Class" if normalize else "Token Count"
        
    plt.figure(figsize=(11, 9))
    sns.heatmap(
        cm_to_plot,
        xticklabels=active_classes,
        yticklabels=active_classes,
        cmap="Blues" if not toxic_only else "YlOrRd",
        annot=True,
        fmt=fmt,
        cbar_kws={"label": cbar_label},
        linewidths=0.5,
        vmin=0.0,
        vmax=1.0 if normalize else None
    )
    subtitle = "Toxic Classes Only" if toxic_only else "All 11 Classes (O at End)"
    plt.title(f"{title} - {subtitle} ({'Normalized' if normalize else 'Counts'})", fontsize=13, pad=12, fontweight="bold")
    plt.xlabel("Predicted Tag", fontsize=11)
    plt.ylabel("True Gold Tag", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path
