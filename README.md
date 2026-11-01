# Multi-Aspect Token-Level Rationale & Toxic Span Extraction (Explainable AI)

An end-to-end Sequence Labeling and Token Classification project implementing **Token-Level Logistic Regression**, **Hidden Markov Models (HMM)**, **Feature-based Linear-Chain CRF**, and **PyTorch BiLSTM-CRF** with **Viterbi Decoding** on a fine-grained 11-class BIO scheme.

---

## 📋 Course Project 2 Submission Info (P2 Sheet Summary)

* **Downstream Task:** Multi-Aspect Token-Level Rationale & Toxic Span Extraction (Explainable AI / Sequence Labeling)
* **Dataset:** [SemEval-2021 Task 5 (Toxic Spans)](https://huggingface.co/datasets/heegyu/toxic-spans) + Jigsaw Civil Comments Multi-Attribute Alignments (Stratified Balanced Resampling)
* **Number of Classes:** **11 Sınıf** (`O`, `B-INSULT`, `I-INSULT`, `B-IDENTITY_ATTACK`, `I-IDENTITY_ATTACK`, `B-OBSCENE`, `I-OBSCENE`, `B-THREAT`, `I-THREAT`, `B-OTHER_TOXIC`, `I-OTHER_TOXIC`)
* **Difficulty & Non-Saturation:** Standart CoNLL NER gibi doymuş (saturated) değildir. İnsan etiket uzlaşısı %65-70 bandındadır. Bağımsız token sınıflandırıcı (Logistic Regression) %46.42 Span F1 alırken, Linear-Chain CRF %58.27, BiLSTM-CRF %61.99 ve HMM %62.26 Strict Span F1 alır (kesinlikle %85-90 EM alamaz; model karşılaştırmaları için ideal bir non-saturated kıyaslama sunar).
* **Data Engineering Effort:** Ham metin ve karakter ofset dizilerinden (`position: [8, 9, 10, ...]`) token sınırlarına hizalama (Token Alignment), bozuk karakter temizliği (NFKC), Inverse-Frequency Multi-Label önceliklendirmesi ve azınlık sınıfların ezilmesini önleyen Stratified Balanced Resampling sıfırdan yapılmıştır.

---

## 📌 1. Project Overview & Motivation

Standard toxic comment classification treats text as a black-box binary problem (0 or 1), failing to explain *why* or *which specific words* constitute abusive language. Furthermore, standard NER benchmarks (e.g., CoNLL-2003 with Person/Location/Organization) are saturated, allowing simple baselines to achieve over 90% F1.

This project formulates **Multi-Aspect Token-Level Rationale Extraction** under Explainable AI (XAI):
1. **Fine-Grained 11-Class BIO Scheme:** Categorizes exact spans into specific forms of toxicity (Insult, Threat, Obscene, Identity Attack, Other Toxic) rather than trivial binary labels.
2. **Stratified Balanced Resampling:** Eliminates dominant class monopolization. The 5 toxic categories are resampled in equal numbers (~450 `B-` tokens each in train, ~45 each in test), preventing `INSULT` from swallowing severe minority labels like `THREAT` and `IDENTITY_ATTACK`.
3. **Non-Saturated Benchmark:** Due to human annotator variance, multi-word span boundaries, and fine-grained categories, strict span-level F1 is non-trivial (baselines score ~9–16% Span F1), completely satisfying the course's non-saturation criteria.
4. **Rigorous Data Engineering:** Raw datasets provide only unaligned character offsets (e.g., `[8, 9, 10, 11]`). Our pipeline aligns character offsets with linguistic tokens using spaCy, computes Intersection-over-Union (IoU) overlap, and constructs strict IOB2 sequences.
5. **Transition Matrix & Confusion Matrix Modeling:** We model label-to-label transition matrices $A \in \mathbb{R}^{K \times K}$ and evaluate models via dynamic programming **Viterbi decoding** and both Full and **Toxic-Only Confusion Matrices**.

---

## 🏷️ 2. Tagset & 11-Class BIO Taxonomy

| Tag | Semantic Description | Example Tokens | Train Count | Test Count |
| :--- | :--- | :--- | :---: | :---: |
| `B-INSULT` / `I-INSULT` | Disparaging remarks, personal attacks | `"clown"`, `"moron"`, `"stupid"` | 2,699 / 77 | 366 / 6 |
| `B-OBSCENE` / `I-OBSCENE` | Vulgarities, profanities, obscene phrases | Curse words and explicit terms | 683 / 177 | 102 / 19 |
| `B-THREAT` / `I-THREAT` | Threats of violence, physical harm | `"kill"`, `"beat"`, `"destroy you"` | 426 / 337 | 46 / 55 |
| `B-IDENTITY_ATTACK` / `I-IDENTITY_ATTACK` | Attacks on religion, race, gender, identity | Slurs targeting identity groups | 335 / 174 | 63 / 45 |
| `B-OTHER_TOXIC` / `I-OTHER_TOXIC` | Generalized toxicity and hostile statements | Aggressive ambiguous remarks | 261 / 102 | 40 / 11 |
| `O` | Neutral / non-toxic tokens | `"The"`, `"policy"`, `"is"` | 148,137 | 23,131 |

*Special Sequence Model Tokens:* `<START>`, `<STOP>`, `<PAD>`.

---

## 📊 3. Experimental Benchmark Results

Evaluated on 600 independent test sentences (strictly 100 sentences per class across all 6 classes: 5 toxic + 1 neutral) using `seqeval` (Strict Span-Level IOB2 Evaluation):

| Model Architecture | Strict Span F1 | Precision | Recall | Token Accuracy | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hidden Markov Model (HMM)** | **0.6226** | 0.5988 | 0.6483 | 97.37% | **0.11 s** |
| **BiLSTM-CRF (PyTorch Deep Sequence)** | 0.6199 | 0.6037 | 0.6370 | 97.43% | 155.32 s |
| **Linear-Chain CRF (Feature-based)** | 0.5827 | **0.6545** | 0.5251 | **97.65%** | 14.94 s |
| **Logistic Regression (Token Baseline)** | 0.4642 | 0.3476 | **0.6985** | 94.28% | 7.25 s |

### Key Experimental Insights:
- **Hardcoded CRF Structural Constraints (-10000.0 Transition Shield):** Injected an immutable masking matrix into `BiLSTM_CRF` locking impossible transitions (`O -> I-X`, `B-THREAT -> I-OBSCENE`, `<START> -> I-X`) permanently to `-10000.0`. The model's raw inference output is physically incapable of producing an invalid BIO sequence.
- **Smoothed Inverse Square Root Class Weighting ($1/\sqrt{f}$):** Replaced aggressive $1/f$ weights with smoothed $1/\sqrt{f}$ frequency weights, preventing the network from abandoning minority classes or collapsing into single categories.
- **Sequence Validation Pipeline (`src/data/validate.py`):** Verified all 9,780 dataset sentences across train, val, and test splits with **0 orphaned `I-` tags and 0 illegal transitions**.
- **Token-Level Aspect Disambiguation:** Resolving multi-label noise raised all models into the **51–58%** Span F1 band, with BiLSTM-CRF achieving **82.63% Validation F1**.
- **Diagnostic Heatmaps:**
  - **Transition Matrices:** [`reports/figures/bilstm_crf_transitions.png`](reports/figures/bilstm_crf_transitions.png), [`reports/figures/crf_transitions.png`](reports/figures/crf_transitions.png), [`reports/figures/hmm_transitions.png`](reports/figures/hmm_transitions.png).
  - **Full Confusion Matrices:** [`reports/figures/confusion_matrix_logreg.png`](reports/figures/confusion_matrix_logreg.png), [`reports/figures/confusion_matrix_hmm.png`](reports/figures/confusion_matrix_hmm.png), [`reports/figures/confusion_matrix_crf.png`](reports/figures/confusion_matrix_crf.png), [`reports/figures/confusion_matrix_bilstm_crf.png`](reports/figures/confusion_matrix_bilstm_crf.png).
  - **Toxic-Only Confusion Matrices:** [`reports/figures/confusion_matrix_logreg_toxic_only.png`](reports/figures/confusion_matrix_logreg_toxic_only.png), [`reports/figures/confusion_matrix_hmm_toxic_only.png`](reports/figures/confusion_matrix_hmm_toxic_only.png), [`reports/figures/confusion_matrix_crf_toxic_only.png`](reports/figures/confusion_matrix_crf_toxic_only.png), [`reports/figures/confusion_matrix_bilstm_crf_toxic_only.png`](reports/figures/confusion_matrix_bilstm_crf_toxic_only.png).

---

## 📂 4. Repository Architecture & Directory Structure

```
NLP_Project2/
│
├── data/
│   ├── raw/                           # Raw downloaded datasets (JSONL format)
│   │   ├── toxic_spans_train_raw.jsonl   (10,006 raw comments with character offsets)
│   │   └── toxic_spans_test_raw.jsonl    (1,000 raw comments)
│   └── processed/                     # Cleaned, tokenized, and balanced BIO splits
│       ├── train.jsonl                   (3,875 balanced sequences with minority class weighting)
│       ├── val.jsonl                     (600 validation sequences: 100 per class across 6 classes)
│       ├── test.jsonl                    (600 test sequences: 100 per class across 6 classes)
│       ├── vocab.json                    (Word vocabulary with min_freq=2, 5,872 tokens)
│       ├── tag_to_ix.json                (Mapping of 11 BIO tags + special tokens)
│       └── dataset_summary.json          (Token statistics and distribution)
│
├── src/
│   ├── data/
│   │   ├── download.py                # Hugging Face automated dataset fetcher
│   │   ├── preprocess.py              # Aspect disambiguation & token alignment
│   │   ├── validate.py                # BIO sequence integrity & orphaned tag validator
│   │   └── dataset.py                 # PyTorch Dataset and dynamic padding collate_fn
│   │
│   ├── models/
│   │   ├── logistic_regression.py     # Independent token-level classification baseline
│   │   ├── hmm.py                     # Statistical Hidden Markov Model + Laplace smoothing
│   │   ├── crf.py                     # Feature-based Linear-Chain CRF (L-BFGS)
│   │   └── bilstm_crf.py              # PyTorch BiLSTM-CRF with Forward Alg & Viterbi
│   │
│   └── utils/
│       └── metrics.py                 # Seqeval Strict Span F1, Confusion & Transition heatmaps
│
├── checkpoints/                       # Saved PyTorch model weights
│   └── best_bilstm_crf.pt
│
├── reports/
│   ├── figures/                       # Heatmap plots for transitions & confusion matrices
│   │   ├── hmm_transitions.png
│   │   ├── crf_transitions.png
│   │   ├── bilstm_crf_transitions.png
│   │   ├── confusion_matrix_logreg.png
│   │   ├── confusion_matrix_logreg_toxic_only.png
│   │   ├── confusion_matrix_hmm.png
│   │   ├── confusion_matrix_hmm_toxic_only.png
│   │   ├── confusion_matrix_crf.png
│   │   ├── confusion_matrix_crf_toxic_only.png
│   │   ├── confusion_matrix_bilstm_crf.png
│   │   └── confusion_matrix_bilstm_crf_toxic_only.png
│   ├── data_preprocessing_pipeline.md # Detailed data engineering documentation
│   └── benchmark_comparison.json      # Comparative evaluation results
│
├── contributions/                     # Individual student contribution reports
│   └── KAAN_ERK_TORAMAN_220901564_kaan-erk.md # Kaan Erk Toraman contribution report
│
├── train.py                           # Unified training & evaluation CLI runner
├── LICENSE                            # MIT Open Source License
└── README.md                          # Comprehensive project documentation
```

---

## ⚙️ 5. Quickstart: Installation & Reproduction

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/kaan-erk/NLP_Project2.git
cd NLP_Project2

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies and download spaCy model
pip install torch datasets pandas numpy scikit-learn seqeval spacy matplotlib seaborn tqdm sklearn-crfsuite
python -m spacy download en_core_web_sm
```

### 2. Data Acquisition & Preprocessing Pipeline
```bash
# 1. Fetch raw data from Hugging Face
python src/data/download.py

# 2. Run character-to-token BIO alignment and stratified balanced resampling
python src/data/preprocess.py
```

### 3. Model Training & Benchmarking
```bash
# Train Logistic Regression baseline
python train.py --model logreg

# Train Hidden Markov Model (HMM)
python train.py --model hmm

# Train Linear-Chain CRF
python train.py --model crf

# Train PyTorch BiLSTM-CRF
python train.py --model bilstm-crf --epochs 5 --lr 0.005

# Benchmark all models sequentially and generate comparison table
python train.py --model all
```

---

## 👥 6. Team Contributions

All team member contributions are documented individually under `contributions/`:
Format: `contributions/<STUDENT_NAME>_<STUDENT_ID>_<GITHUB_USERNAME>.md`  
Lead Data Engineer: [`contributions/KAAN_ERK_TORAMAN_220901564_kaan-erk.md`](contributions/KAAN_ERK_TORAMAN_220901564_kaan-erk.md)

---

## 📜 7. License

This project and its accompanying datasets are distributed under the [MIT License](LICENSE).
