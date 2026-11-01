# Contribution Report

* **Student Name:** Burak Gizir

* **Student ID:** 220901544

* **GitHub Username:** burakgizir-01

## 1. Overview of Contributions

In this project, I was responsible for orchestrating the main model training pipeline, resolving severe class imbalance issues during model optimization, and engineering the comprehensive evaluation and benchmarking suite. My primary deliverables included the centralized training script, dynamic runtime data balancing (oversampling/downsampling), aggressive loss weighting algorithms, and the final visualization generation for model comparison.

## 2. Specific Modules & Code Implemented

Detailed breakdown of components designed, written, or debugged by me:

### A. Main Training Orchestration (`train.py`)

* **Centralized Pipeline:** Developed the `train.py` execution script to dynamically instantiate, train, and evaluate Logistic Regression, HMM, Linear-Chain CRF, and PyTorch BiLSTM-CRF models from a single interface.

* **BiLSTM-CRF Optimization Loop:** Wrote the PyTorch training loop incorporating gradient clipping, AdamW optimization, and early stopping based on Strict Span F1 validation scores.

* **Gradient Freezing Mechanism:** Integrated a safety block to manually zero-out gradients for illegal BIO transitions during backpropagation, preventing the optimizer from overriding the CRF's structural rules.

### B. Runtime Data Balancing & Imbalance Resolution (`train.py`)

* **Stochastic Downsampling:** Engineered a runtime dataset balancing block that randomly drops 50% of background sentences containing only neutral (`O`) tags to prevent the majority class from acting as a "black hole" during neural network training.

* **Minority Class Oversampling:** Implemented strategic duplication (2x multiplier) for highly imbalanced toxic spans (`B-THREAT`, `B-IDENTITY_ATTACK`) immediately prior to `DataLoader` batching to prevent model memorization while boosting minority recall.

* **Aggressive Inverse Frequency Weighting:** Calculated and applied smoothed inverse-frequency class weights dynamically mapped to the device, forcing the PyTorch loss function to heavily penalize errors on rare toxic categories.

### C. Evaluation & Benchmarking (`src/utils/metrics.py` & `plot_metrics.py`)

* **Confusion Matrix Generation:** Programmed the automated saving of token-level confusion matrix heatmaps, designing both "Full 11-Class" and "Toxic-Classes Only" variations to accurately diagnose performance without the `O` class masking results.

* **Benchmark Data Extraction:** Automated the extraction and JSON serialization of Strict Span-Level metrics (Precision, Recall, F1, Accuracy) and training durations for all models.

* **Visual Analytics:** Authored `plot_metrics.py` to parse the benchmark JSON and generate high-resolution grouped bar charts comparing the four distinct sequence labeling algorithms across multiple evaluation metrics.

## 3. Key Pull Requests / Git Commits

List of key commits / PRs contributed to the repository:

* `commit [hash]`: Added centralized `train.py` for model orchestration and automated evaluation logging

* `commit [hash]`: Implemented runtime data balancing (O-tag downsampling & rare tag oversampling) to resolve BiLSTM-CRF class collapse

* `commit [hash]`: Added Aggressive Inverse Frequency tensor weighting for the BiLSTM-CRF objective function

* `commit [hash]`: Created `plot_metrics.py` and finalized the confusion matrix generation pipelines for the benchmark report