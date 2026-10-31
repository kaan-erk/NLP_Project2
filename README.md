# Multi-Aspect Token-Level Rationale & Toxic Span Extraction (Explainable AI)

An end-to-end Sequence Labeling and Token Classification project implementing **Token-Level Logistic Regression**, **Hidden Markov Models (HMM)**, **Feature-based Linear-Chain CRF**, and **PyTorch BiLSTM-CRF** with **Viterbi Decoding** on a fine-grained 11-class BIO scheme.

---

## 📋 Course Project 2 Submission Info (P2 Sheet Summary)

* **Downstream Task:** Multi-Aspect Token-Level Rationale & Toxic Span Extraction (Explainable AI)
* **Dataset:** [SemEval-2021 Task 5 (Toxic Spans)](https://huggingface.co/datasets/heegyu/toxic-spans) + Jigsaw Civil Comments Multi-Attribute Alignments
* **Number of Classes:** **11 Sınıf** (`O`, `B-INSULT`, `I-INSULT`, `B-IDENTITY_ATTACK`, `I-IDENTITY_ATTACK`, `B-OBSCENE`, `I-OBSCENE`, `B-THREAT`, `I-THREAT`, `B-OTHER_TOXIC`, `I-OTHER_TOXIC`)
* **Difficulty & Non-Saturation:** Standart CoNLL NER gibi doymuş (saturated) değildir. İnsan etiket uzlaşısı %65-70 bandındadır. Bağımsız token sınıflandırıcı (Logistic Regression) yalnızca %16.79 Span F1, HMM baseline %35.41 Span F1 alırken, derin BiLSTM-CRF modeli %55.78 F1 skoruna ulaşır (kesinlikle %85-90 EM alamaz).
* **Data Engineering Effort:** Ham metin ve karakter ofset dizilerinden (`position: [8, 9, 10, ...]`) token sınırlarına hizalama (Token Alignment), bozuk karakter temizliği (NFKC) ve BIO etiket üretimi sıfırdan yapılmıştır.

---

## 📌 1. Project Overview & Motivation

Standard toxic comment classification treats text as a black-box binary problem (0 or 1), failing to explain *why* or *which specific words* constitute abusive language. Furthermore, standard NER benchmarks (e.g., CoNLL-2003 with Person/Location/Organization) are saturated, allowing simple baselines to achieve over 90% F1.

This project formulates **Multi-Aspect Token-Level Rationale Extraction** under Explainable AI (XAI):
1. **Fine-Grained 11-Class BIO Scheme:** Categorizes exact spans into specific forms of toxicity (Insult, Threat, Obscene, Identity Attack, Other Toxic) rather than trivial binary labels.
2. **Non-Saturated Benchmark:** Due to human annotator variance and fuzzy span boundaries, strict span-level F1 is non-trivial (baselines score ~16–45% Span F1), completely satisfying the non-saturation criteria.
3. **Rigorous Data Engineering:** Raw datasets provide only unaligned character offsets (e.g., `[8, 9, 10, 11]`). Our pipeline aligns character offsets with linguistic tokens using spaCy, computes Intersection-over-Union (IoU) overlap, and constructs strict IOB2 sequences.
4. **Transition Matrix & Confusion Matrix Modeling:** We model label-to-label transition matrices $A \in \mathbb{R}^{K \times K}$ and evaluate models via dynamic programming **Viterbi decoding** and token-level **Confusion Matrices**.

---

## 🏷️ 2. Tagset & 11-Class BIO Taxonomy

| Tag | Semantic Description | Example Tokens |
| :--- | :--- | :--- |
| `O` | Neutral / non-toxic tokens | `"The"`, `"policy"`, `"is"` |
| `B-INSULT` / `I-INSULT` | Disparaging remarks, personal attacks | `"clown"`, `"moron"`, `"stupid"` |
| `B-IDENTITY_ATTACK` / `I-IDENTITY_ATTACK` | Attacks on religion, race, gender, identity | Slurs targeting identity groups |
| `B-OBSCENE` / `I-OBSCENE` | Vulgarities, profanities, obscene phrases | Curse words and explicit terms |
| `B-THREAT` / `I-THREAT` | Threats of violence, physical harm | `"kill"`, `"beat"`, `"destroy you"` |
| `B-OTHER_TOXIC` / `I-OTHER_TOXIC` | Generalized toxicity and hostile statements | Aggressive ambiguous remarks |

*Special Sequence Model Tokens:* `<START>`, `<STOP>`, `<PAD>`.

---

## 📊 3. Experimental Benchmark Results

Evaluated on 1,000 independent test sentences using `seqeval` (Strict Span-Level IOB2 Evaluation):

| Model Architecture | Strict Span F1 | Precision | Recall | Token Accuracy | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Token Baseline)** | **0.1679** | 0.1010 | 0.4983 | 87.62% | 10.41 s |
| **Hidden Markov Model (HMM)** | **0.3541** | 0.3132 | 0.4073 | 96.12% | 0.18 s |
| **Linear-Chain CRF (Feature-based)** | **0.4525** | 0.5359 | 0.3916 | 98.58% | 20.62 s |
| **BiLSTM-CRF (PyTorch Deep Sequence)** | **0.5578** | 0.5255 | 0.5944 | 98.66% | 268.20 s |

### Key Experimental Insights:
- **Sequential vs Non-Sequential Modeling:** Logistic Regression achieves only **16.79% Span F1** because it classifies tokens independently and frequently violates BIO grammar rules. Adding sequential transition modeling in HMM boosts Span F1 by **+18.62%**.
- **Discriminative Feature Engineering:** Linear-Chain CRF incorporates orthographic, casing, POS tags, and context windows, raising Span F1 to **45.25%** (+9.84% over HMM).
- **Deep Sequence Modeling:** BiLSTM-CRF achieves the benchmark peak at **55.78% Strict Span F1** with **59.44% Recall**, outperforming the baseline by **+38.99% absolute**.
- **Heatmaps & Diagnostics:**
  - **Transition Matrices:** [`reports/figures/bilstm_crf_transitions.png`](reports/figures/bilstm_crf_transitions.png), [`reports/figures/crf_transitions.png`](reports/figures/crf_transitions.png), [`reports/figures/hmm_transitions.png`](reports/figures/hmm_transitions.png).
  - **Confusion Matrices:** [`reports/figures/confusion_matrix_bilstm_crf.png`](reports/figures/confusion_matrix_bilstm_crf.png), [`reports/figures/confusion_matrix_crf.png`](reports/figures/confusion_matrix_crf.png), [`reports/figures/confusion_matrix_hmm.png`](reports/figures/confusion_matrix_hmm.png), [`reports/figures/confusion_matrix_logreg.png`](reports/figures/confusion_matrix_logreg.png).

---

## 📂 4. Repository Architecture & Directory Structure

```
NLP_Project2/
│
├── data/
│   ├── raw/                           # Raw downloaded datasets (JSONL format)
│   │   ├── toxic_spans_train_raw.jsonl   (10,006 raw comments with character offsets)
│   │   └── toxic_spans_test_raw.jsonl    (1,000 raw comments)
│   └── processed/                     # Cleaned, tokenized, and aligned BIO splits
│       ├── train.jsonl                   (8,506 aligned training sequences)
│       ├── val.jsonl                     (1,500 validation sequences)
│       ├── test.jsonl                    (1,000 test sequences)
│       ├── vocab.json                    (Word vocabulary with min_freq=2)
│       ├── tag_to_ix.json                (Mapping of 11 BIO tags + special tokens)
│       └── dataset_summary.json          (Token statistics and distribution)
│
├── src/
│   ├── data/
│   │   ├── download.py                # Hugging Face automated dataset fetcher
│   │   ├── preprocess.py              # Character-to-token BIO alignment & sanitization
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
│   │   ├── confusion_matrix_hmm.png
│   │   ├── confusion_matrix_crf.png
│   │   └── confusion_matrix_bilstm_crf.png
│   ├── data_preprocessing_pipeline.md # Detailed data engineering documentation
│   └── benchmark_comparison.json      # Comparative evaluation results
│
├── contributions/                     # Individual student contribution reports
│   └── TEMPLATE.md                    # Template for team member reports
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
git clone https://github.com/<YOUR_REPO>/NLP_Project2.git
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

# 2. Run character-to-token BIO alignment and generate splits
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
*(Example: `contributions/HASAN_KAYA_230123123_hasankayyya.md`)*  
See [`contributions/TEMPLATE.md`](contributions/TEMPLATE.md) for instructions.

---

## 📜 7. License

This project and its accompanying datasets are distributed under the [MIT License](LICENSE).
