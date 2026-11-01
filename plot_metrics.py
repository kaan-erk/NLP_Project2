import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
REPORTS_DIR = Path("reports")
JSON_PATH = REPORTS_DIR / "benchmark_comparison.json"
OUTPUT_PATH = REPORTS_DIR / "figures" / "benchmark_metrics_comparison.png"

def plot_all_metrics():
    if not JSON_PATH.exists():
        print(f"[ERROR] {JSON_PATH} not found. Ensure train.py has successfully finished running.")
        return

    # Load the metrics exported by train.py
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract data and simplify model names for the X-axis
    models = [item["Model"].split(" ")[0] for item in data] 
    precisions = [item["Precision"] for item in data]
    recalls = [item["Recall"] for item in data]
    f1_scores = [item["Strict Span F1"] for item in data]
    accuracies = [item["Token Accuracy"] for item in data]

    # Set up the grouped bar chart dimensions
    x = np.arange(len(models))
    width = 0.2  

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each metric aspect
    rects1 = ax.bar(x - 1.5*width, precisions, width, label='Precision', color='#4C72B0')
    rects2 = ax.bar(x - 0.5*width, recalls, width, label='Recall', color='#DD8452')
    rects3 = ax.bar(x + 0.5*width, f1_scores, width, label='Strict Span F1', color='#55A868')
    rects4 = ax.bar(x + 1.5*width, accuracies, width, label='Token Accuracy', color='#C44E52')

    # Formatting and labels
    ax.set_ylabel('Score (0.0 to 1.0)')
    ax.set_title('Token Classification Benchmark Metrics by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight='bold')
    ax.set_ylim(0, 1.15) 
    ax.legend(loc='upper right')

    # Attach raw score labels directly above each bar
    for rects in [rects1, rects2, rects3, rects4]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    # Save to the figures directory
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    print(f"[SUCCESS] Grouped metrics comparison graph saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    plot_all_metrics()
