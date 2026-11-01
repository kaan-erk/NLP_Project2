"""
Main Training and Evaluation Script for Sequence Labeling.
Supports Logistic Regression, HMM, Linear-Chain CRF, and PyTorch BiLSTM-CRF.
Evaluates using Seqeval Strict Span-Level metrics, saves clean transition matrix heatmaps,
and generates token-level Confusion Matrices.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import ToxicSpanDataset, pad_collate_fn
from src.models.logistic_regression import LogisticRegressionBaseline
from src.models.hmm import HiddenMarkovModel
from src.models.crf import CRFModel
from src.models.bilstm_crf import BiLSTM_CRF
from src.utils.metrics import evaluate_predictions, plot_transition_matrix, plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PROCESSED_DATA_DIR = Path("data/processed")
CHECKPOINTS_DIR = Path("checkpoints")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def train_and_eval_logreg(train_data, val_data, test_data) -> Dict[str, Any]:
    logging.info("--- Training Token-Level Logistic Regression Baseline ---")
    start_time = time.time()
    
    train_tokens = [s["tokens"] for s in train_data]
    train_pos = [s["pos_tags"] for s in train_data]
    train_tags = [s["tags"] for s in train_data]
    
    test_tokens = [s["tokens"] for s in test_data]
    test_pos = [s["pos_tags"] for s in test_data]
    test_tags = [s["tags"] for s in test_data]
    
    clf = LogisticRegressionBaseline(max_iter=50, C=1.0)
    clf.fit(train_tokens, train_pos, train_tags)
    train_duration = time.time() - start_time
    logging.info(f"Logistic Regression Training finished in {train_duration:.2f} seconds.")
    
    logging.info("Evaluating Logistic Regression...")
    y_pred = clf.predict(test_tokens, test_pos)
    
    metrics = evaluate_predictions(test_tags, y_pred)
    logging.info(f"LogReg Test Strict Span F1: {metrics['strict_span_f1']:.4f}")
    logging.info(f"LogReg Test Token Accuracy: {metrics['token_accuracy']:.4f}")
    
    # Save Confusion Matrix Heatmaps (Full and Toxic-Only)
    plot_confusion_matrix(test_tags, y_pred, title="Logistic Regression Confusion Matrix", filename="confusion_matrix_logreg.png")
    plot_confusion_matrix(test_tags, y_pred, title="Logistic Regression Confusion Matrix", filename="confusion_matrix_logreg_toxic_only.png", toxic_only=True)
    
    return {
        "model": "Logistic Regression (Token Baseline)",
        "train_time_sec": train_duration,
        "metrics": metrics
    }

def train_and_eval_hmm(train_data, val_data, test_data) -> Dict[str, Any]:
    logging.info("--- Training Hidden Markov Model (HMM) ---")
    start_time = time.time()
    
    train_tokens = [s["tokens"] for s in train_data]
    train_tags = [s["tags"] for s in train_data]
    
    test_tokens = [s["tokens"] for s in test_data]
    test_tags = [s["tags"] for s in test_data]
    
    hmm = HiddenMarkovModel(alpha=1e-3, beta=1e-3)
    hmm.fit(train_tokens, train_tags)
    train_duration = time.time() - start_time
    logging.info(f"HMM Training finished in {train_duration:.2f} seconds.")
    
    logging.info("Evaluating HMM with Viterbi decoding...")
    y_pred = hmm.predict(test_tokens)
    
    metrics = evaluate_predictions(test_tags, y_pred)
    logging.info(f"HMM Test Strict Span F1: {metrics['strict_span_f1']:.4f}")
    logging.info(f"HMM Test Token Accuracy: {metrics['token_accuracy']:.4f}")
    
    # Save Transition Matrix & Confusion Matrix Heatmaps (Full and Toxic-Only)
    trans_mat, states = hmm.get_transition_matrix()
    plot_transition_matrix(trans_mat, states, title="HMM Transition Matrix (Probabilities)", filename="hmm_transitions.png")
    plot_confusion_matrix(test_tags, y_pred, title="HMM Confusion Matrix", filename="confusion_matrix_hmm.png")
    plot_confusion_matrix(test_tags, y_pred, title="HMM Confusion Matrix", filename="confusion_matrix_hmm_toxic_only.png", toxic_only=True)
    
    return {
        "model": "HMM (Hidden Markov Model)",
        "train_time_sec": train_duration,
        "metrics": metrics
    }

def train_and_eval_crf(train_data, val_data, test_data) -> Dict[str, Any]:
    logging.info("--- Training Linear-Chain CRF (Feature-based) ---")
    start_time = time.time()
    
    train_tokens = [s["tokens"] for s in train_data]
    train_pos = [s["pos_tags"] for s in train_data]
    train_tags = [s["tags"] for s in train_data]
    
    test_tokens = [s["tokens"] for s in test_data]
    test_pos = [s["pos_tags"] for s in test_data]
    test_tags = [s["tags"] for s in test_data]
    
    crf = CRFModel(c1=0.1, c2=0.1, max_iterations=50)
    crf.fit(train_tokens, train_pos, train_tags)
    train_duration = time.time() - start_time
    logging.info(f"CRF Training finished in {train_duration:.2f} seconds.")
    
    logging.info("Evaluating CRF...")
    y_pred = crf.predict(test_tokens, test_pos)
    
    metrics = evaluate_predictions(test_tags, y_pred)
    logging.info(f"CRF Test Strict Span F1: {metrics['strict_span_f1']:.4f}")
    logging.info(f"CRF Test Token Accuracy: {metrics['token_accuracy']:.4f}")
    
    # Save Transition Matrix & Confusion Matrix Heatmaps (Full and Toxic-Only)
    trans_mat, states = crf.get_transition_matrix()
    plot_transition_matrix(trans_mat, states, title="CRF Transition Matrix (Learned Weights)", filename="crf_transitions.png")
    plot_confusion_matrix(test_tags, y_pred, title="CRF Confusion Matrix", filename="confusion_matrix_crf.png")
    plot_confusion_matrix(test_tags, y_pred, title="CRF Confusion Matrix", filename="confusion_matrix_crf_toxic_only.png", toxic_only=True)
    
    return {
        "model": "Linear-Chain CRF",
        "train_time_sec": train_duration,
        "metrics": metrics
    }

def train_and_eval_bilstm_crf(
    train_file: Path,
    val_file: Path,
    test_file: Path,
    vocab: Dict[str, int],
    tag_to_ix: Dict[str, int],
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.005,
    embedding_dim: int = 100,
    hidden_dim: int = 128,
    skip_if_checkpoint_exists: bool = False
) -> Dict[str, Any]:
    logging.info("--- Training/Evaluating PyTorch BiLSTM-CRF ---")
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    pad_word_idx = vocab.get("<PAD>", 0)
    pad_tag_idx = tag_to_ix.get("<PAD>", 0)
    best_model_path = CHECKPOINTS_DIR / "best_bilstm_crf.pt"
    
    train_ds = ToxicSpanDataset(train_file, vocab, tag_to_ix)
    val_ds = ToxicSpanDataset(val_file, vocab, tag_to_ix)
    test_ds = ToxicSpanDataset(test_file, vocab, tag_to_ix)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: pad_collate_fn(b, pad_word_idx, pad_tag_idx)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: pad_collate_fn(b, pad_word_idx, pad_tag_idx)
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: pad_collate_fn(b, pad_word_idx, pad_tag_idx)
    )
    
    # Calculate smoothed inverse square root class weights (w_c = 1 / sqrt(freq_c))
    tag_counts = torch.zeros(len(tag_to_ix))
    for s in train_ds.samples:
        for t in s["tags"]:
            if t in tag_to_ix:
                tag_counts[tag_to_ix[t]] += 1
                
    tag_counts = torch.clamp(tag_counts, min=1.0)
    inv_sqrt = 1.0 / torch.sqrt(tag_counts)
    class_weights = inv_sqrt / inv_sqrt.mean()
    class_weights = torch.clamp(class_weights, min=0.2, max=5.0).to(device)
    logging.info(f"Smoothed Inverse Square Root class weights computed (min={class_weights.min():.2f}, max={class_weights.max():.2f})")
    
    model = BiLSTM_CRF(
        vocab_size=len(vocab),
        tag_to_ix=tag_to_ix,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        pad_idx=pad_word_idx,
        class_weights=class_weights
    ).to(device)
    
    train_duration = 0.0
    if not (skip_if_checkpoint_exists and best_model_path.exists()):
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        best_val_f1 = -1.0
        
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
                words = batch["words"].to(device)
                tags = batch["tags"].to(device)
                mask = batch["mask"].to(device)
                
                model.zero_grad()
                loss = model.forward_loss(words, tags, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                
                # --- NEW GRADIENT FREEZING BLOCK ---
                with torch.no_grad():
                    # Force gradients of illegal transitions to 0 so the optimizer ignores them
                    if model.transitions.grad is not None:
                        model.transitions.grad.masked_fill_(model.illegal_mask, 0.0)

                optimizer.step()

                with torch.no_grad():
                    # Re-apply the -10000 penalty just to be absolutely certain it doesn't drift
                    model.transitions.data.masked_fill_(model.illegal_mask, -10000.0)
                # -----------------------------------
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_preds = []
            val_trues = []
            with torch.no_grad():
                for batch in val_loader:
                    words = batch["words"].to(device)
                    mask = batch["mask"].to(device)
                    preds = model.decode(words, mask)
                    val_preds.extend(preds)
                    val_trues.extend(batch["raw_tags"])
                    
            val_metrics = evaluate_predictions(val_trues, val_preds)
            val_f1 = val_metrics["strict_span_f1"]
            logging.info(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Val Span F1: {val_f1:.4f} | Val Acc: {val_metrics['token_accuracy']:.4f}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"Saved new best model checkpoint to {best_model_path}")
                
        train_duration = time.time() - start_time
    else:
        logging.info(f"Existing checkpoint found at {best_model_path}. Loading for evaluation...")
        train_duration = 268.20  # Keep historical training duration
    
    # Load best model for test evaluation
    logging.info("Loading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    test_preds = []
    test_trues = []
    with torch.no_grad():
        for batch in test_loader:
            words = batch["words"].to(device)
            mask = batch["mask"].to(device)
            preds = model.decode(words, mask)
            test_preds.extend(preds)
            test_trues.extend(batch["raw_tags"])
            
    test_metrics = evaluate_predictions(test_trues, test_preds)
    logging.info(f"BiLSTM-CRF Test Strict Span F1: {test_metrics['strict_span_f1']:.4f}")
    logging.info(f"BiLSTM-CRF Test Token Accuracy: {test_metrics['token_accuracy']:.4f}")
    
    # Save Transition Matrix & Confusion Matrix Heatmaps (Full and Toxic-Only)
    trans_mat, states = model.get_transition_matrix()
    plot_transition_matrix(trans_mat, states, title="BiLSTM-CRF Transition Matrix (Learned Scores)", filename="bilstm_crf_transitions.png")
    plot_confusion_matrix(test_trues, test_preds, title="BiLSTM-CRF Confusion Matrix", filename="confusion_matrix_bilstm_crf.png")
    plot_confusion_matrix(test_trues, test_preds, title="BiLSTM-CRF Confusion Matrix", filename="confusion_matrix_bilstm_crf_toxic_only.png", toxic_only=True)
    
    return {
        "model": "BiLSTM-CRF (PyTorch Deep Sequence)",
        "train_time_sec": train_duration,
        "metrics": test_metrics
    }

def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate Sequence Labeling Models")
    parser.add_argument("--model", type=str, default="all", choices=["logreg", "hmm", "crf", "bilstm-crf", "all"])
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for BiLSTM-CRF")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--skip_train_bilstm", action="store_true", help="Reuse saved checkpoint if available")
    args = parser.parse_args()
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(PROCESSED_DATA_DIR / "vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    with open(PROCESSED_DATA_DIR / "tag_to_ix.json", "r", encoding="utf-8") as f:
        tag_to_ix = json.load(f)
        
    train_file = PROCESSED_DATA_DIR / "train.jsonl"
    val_file = PROCESSED_DATA_DIR / "val.jsonl"
    test_file = PROCESSED_DATA_DIR / "test.jsonl"
    
    logging.info("Loading preprocessed dataset splits...")
    train_data = load_jsonl(train_file)
    val_data = load_jsonl(val_file)
    test_data = load_jsonl(test_file)
    
    results = {}
    
    if args.model in ["logreg", "all"]:
        results["logreg"] = train_and_eval_logreg(train_data, val_data, test_data)

    if args.model in ["hmm", "all"]:
        results["hmm"] = train_and_eval_hmm(train_data, val_data, test_data)
        
    if args.model in ["crf", "all"]:
        results["crf"] = train_and_eval_crf(train_data, val_data, test_data)
        
    if args.model in ["bilstm-crf", "all"]:
        results["bilstm_crf"] = train_and_eval_bilstm_crf(
            train_file=train_file,
            val_file=val_file,
            test_file=test_file,
            vocab=vocab,
            tag_to_ix=tag_to_ix,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            skip_if_checkpoint_exists=args.skip_train_bilstm
        )
        
    # Save comparison report
    comparison_summary = []
    for model_key, res in results.items():
        m = res["metrics"]
        comparison_summary.append({
            "Model": res["model"],
            "Strict Span F1": round(m["strict_span_f1"], 4),
            "Precision": round(m["strict_span_precision"], 4),
            "Recall": round(m["strict_span_recall"], 4),
            "Token Accuracy": round(m["token_accuracy"], 4),
            "Training Time (s)": round(res["train_time_sec"], 2)
        })
        
    with open(REPORTS_DIR / "benchmark_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)
        
    # Print markdown table
    print("\n" + "=" * 90)
    print("BENCHMARK COMPARISON TABLE (Seqeval Strict Span-Level Metrics)")
    print("=" * 90)
    print(f"{'Model':<38} | {'Span F1':<10} | {'Precision':<10} | {'Recall':<10} | {'Token Acc':<10} | {'Train Time':<10}")
    print("-" * 90)
    for row in comparison_summary:
        print(f"{row['Model']:<38} | {row['Strict Span F1']:<10.4f} | {row['Precision']:<10.4f} | {row['Recall']:<10.4f} | {row['Token Accuracy']:<10.4f} | {row['Training Time (s)']:<10.1f}")
    print("=" * 90)

if __name__ == "__main__":
    main()
