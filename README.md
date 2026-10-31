# Multi-Aspect Token-Level Rationale & Toxic Span Extraction (Explainable AI)

An end-to-end Sequence Labeling and Token Classification project implementing **Hidden Markov Models (HMM)**, **Feature-based Linear-Chain CRF**, and **PyTorch BiLSTM-CRF** with **Viterbi Decoding** on a fine-grained 11-class BIO scheme.

---

## 📋 Course Project 2 Submission Info (P2 Sheet Summary)

* **Downstream Task:** Multi-Aspect Token-Level Rationale & Toxic Span Extraction (Explainable AI)
* **Dataset:** [SemEval-2021 Task 5 (Toxic Spans)](https://huggingface.co/datasets/heegyu/toxic-spans) + Jigsaw Civil Comments Multi-Attribute Alignments
* **Number of Classes:** **11 Sınıf** (`O`, `B-INSULT`, `I-INSULT`, `B-IDENTITY_ATTACK`, `I-IDENTITY_ATTACK`, `B-OBSCENE`, `I-OBSCENE`, `B-THREAT`, `I-THREAT`, `B-OTHER_TOXIC`, `I-OTHER_TOXIC`)
* **Difficulty & Non-Saturation:** Standart CoNLL NER gibi doymuş (saturated) değildir. İnsan etiket uzlaşısı (inter-annotator agreement) %65-70 bandındadır. Baseline modeller %35-45 F1 bandında kalır; derin BiLSTM-CRF modeli %55.78 F1 skoruna ulaşır (kesinlikle %85-90 EM alamaz).
* **Data Engineering Effort:** Ham metin ve karakter ofset dizilerinden (`position: [8, 9, 10, ...]`) token sınırlarına hizalama (Token Alignment), bozuk karakter temizliği (NFKC) ve BIO etiket üretimi sıfırdan yapılmıştır.

---

## 📌 1. Project Overview & Motivation

Standard toxic comment classification treats text as a black-box binary problem (0 or 1), failing to explain *why* or *which specific words* constitute abusive language. Furthermore, standard NER benchmarks (e.g., CoNLL-2003 with Person/Location/Organization) are saturated, allowing simple baselines to achieve over 90% F1.

This project formulates **Multi-Aspect Token-Level Rationale Extraction** under Explainable AI (XAI):
1. **Fine-Grained 11-Class BIO Scheme:** Categorizes exact spans into specific forms of toxicity (Insult, Threat, Obscene, Identity Attack, Other Toxic) rather than trivial binary labels.
2. **Non-Saturated Benchmark:** Due to human annotator variance and fuzzy span boundaries, strict span-level F1 is non-trivial (baselines score ~35–45% Span F1), completely satisfying the non-saturation criteria.
3. **Rigorous Data Engineering:** Raw datasets provide only unaligned character offsets (e.g., `[8, 9, 10, 11]`). Our pipeline aligns character offsets with linguistic tokens using spaCy, computes Intersection-over-Union (IoU) overlap, and constructs strict IOB2 sequences.
4. **Direct Transition Matrix Modeling:** Both statistical (HMM) and discriminative (CRF, BiLSTM-CRF) models explicitly model label-to-label transition matrices $A \in \mathbb{R}^{K \times K}$ and resolve valid paths via the dynamic programming **Viterbi algorithm**.

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
| **Hidden Markov Model (HMM)** | **0.3541** | 0.3132 | 0.4073 | 96.12% | 0.18 s |
| **Linear-Chain CRF (Feature-based)** | **0.4447** | 0.5443 | 0.3759 | 98.57% | 17.28 s |
| **BiLSTM-CRF (PyTorch Deep Sequence)** | **0.5578** | 0.5255 | 0.5944 | 98.66% | 268.20 s |

### Key Experimental Insights:
- **Baseline Non-Saturation:** The HMM baseline scores **35.41% Span F1**, verifying that the task is challenging and far from saturated.
- **Feature Engineering Lift:** Handcrafted orthographic features (affixes, casing, POS tags, token context windows) boost CRF performance by **+9.06%** over HMM.
- **Deep Sequence Modeling:** BiLSTM-CRF achieves the highest Strict Span F1 (**55.78%**), demonstrating a **+20.37% absolute improvement** over HMM.
- **Transition Matrix Heatmaps:** Generated transition matrices are saved under `reports/figures/` (`hmm_transitions.png`, `crf_transitions.png`, `bilstm_crf_transitions.png`), demonstrating that the CRF layers naturally penalize invalid transitions (e.g., $P(\text{I-THREAT} \mid \text{B-INSULT}) \approx -\infty$).

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
│   │   ├── hmm.py                     # Statistical Hidden Markov Model + Laplace smoothing
│   │   ├── crf.py                     # Feature-based Linear-Chain CRF (L-BFGS)
│   │   └── bilstm_crf.py              # PyTorch BiLSTM-CRF with Forward Alg & Viterbi
│   │
│   └── utils/
│       └── metrics.py                 # Seqeval Strict Span F1 & Transition heatmaps
│
├── checkpoints/                       # Saved PyTorch model weights
│   └── best_bilstm_crf.pt
│
├── reports/
│   ├── figures/                       # Transition matrix heatmap plots
│   │   ├── hmm_transitions.png
│   │   ├── crf_transitions.png
│   │   └── bilstm_crf_transitions.png
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
# Train and evaluate Hidden Markov Model (HMM)
python train.py --model hmm

# Train and evaluate Linear-Chain CRF
python train.py --model crf

# Train and evaluate PyTorch BiLSTM-CRF
python train.py --model bilstm-crf --epochs 5 --lr 0.005

# Benchmark all models sequentially and generate comparison table
python train.py --model all
```

---

## 🔬 6. Model Architecture & Mathematical Formulation

### 1. Hidden Markov Model (HMM)
- **Transition Probabilities:**
  $$P(y_t \mid y_{t-1}) = \frac{C(y_{t-1}, y_t) + \alpha}{C(y_{t-1}) + \alpha |S|}$$
- **Emission Probabilities:**
  $$P(x_t \mid y_t) = \frac{C(y_t, x_t) + \beta}{C(y_t) + \beta |V|}$$
- **Inference:** Exact dynamic programming via the Viterbi algorithm.

### 2. Feature-based Linear-Chain CRF
Models conditional distribution $P(y \mid x) = \frac{1}{Z(x)} \exp \left( \sum_{t=1}^T \sum_k w_k f_k(y_{t-1}, y_t, x, t) \right)$ using handcrafted orthographic features (prefixes, suffixes, casing, punctuation), part-of-speech (POS) tags, and contextual token windows ($t-1$, $t+1$).

### 3. PyTorch BiLSTM-CRF
- **BiLSTM Representation:** Projects token embeddings through forward and backward LSTM cells to produce contextual emissions $P_{i, y_i}$.
- **CRF Transition Layer:** Models label transition matrix $A_{j, i}$ directly as learnable parameters.
- **Partition Function:** Evaluated via the Forward algorithm in log-space using the numerically stable `log_sum_exp` operator:
  $$\log Z(x) = \log \sum_{y'} \exp(\text{Score}(x, y'))$$
- **Objective:** Minimized via Negative Log-Likelihood loss:
  $$\mathcal{L}_{\text{NLL}} = \log Z(x) - \text{Score}(x, y)$$
- **Decoding:** Exact optimal sequence decoding via Viterbi backtracking.

---

## 👥 7. Team Contributions

All team member contributions are documented individually under `contributions/`:
Format: `contributions/<STUDENT_NAME>_<STUDENT_ID>_<GITHUB_USERNAME>.md`  
*(Example: `contributions/HASAN_KAYA_230123123_hasankayyya.md`)*  
See [`contributions/TEMPLATE.md`](contributions/TEMPLATE.md) for instructions.

---

## 📜 8. License

This project and its accompanying datasets are distributed under the [MIT License](LICENSE).
