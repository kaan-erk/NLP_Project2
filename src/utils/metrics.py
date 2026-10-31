"""
Evaluation and visualization utilities for sequence labeling.
Computes seqeval Span-Level F1/Precision/Recall and plots CRF transition heatmaps.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from seqeval.metrics import (
    classification_report as seqeval_report,
    f1_score as seqeval_f1,
    precision_score as seqeval_precision,
    recall_score as seqeval_recall,
)
from seqeval.scheme import IOB2

FIGURES_DIR = Path("reports/figures")

def evaluate_predictions(y_true: List[List[str]], y_pred: List[List[str]]) -> Dict[str, Any]:
    """
    Computes strict span-level metrics using seqeval (IOB2 scheme).
    """
    f1 = seqeval_f1(y_true, y_pred, mode="strict", scheme=IOB2)
    prec = seqeval_precision(y_true, y_pred, mode="strict", scheme=IOB2)
    rec = seqeval_recall(y_true, y_pred, mode="strict", scheme=IOB2)
    report = seqeval_report(y_true, y_pred, mode="strict", scheme=IOB2, digits=4)
    
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
    filename: str = "transition_matrix.png"
):
    """
    Plots and saves a heatmap of transition probabilities/scores between tags.
    Essential for demonstrating linguistic constraints learned by HMM/CRF models.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / filename
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix, 
        xticklabels=labels, 
        yticklabels=labels, 
        cmap="coolwarm", 
        annot=True, 
        fmt=".2f",
        cbar=True
    )
    plt.title(title, fontsize=14, pad=12)
    plt.xlabel("To Tag (y_t)", fontsize=12)
    plt.ylabel("From Tag (y_{t-1})", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Transition matrix heatmap saved to: {out_path}")
    return out_path

