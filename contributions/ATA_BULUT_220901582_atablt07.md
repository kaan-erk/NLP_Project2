# Contribution Report: Ata Bulut

- **Student Name:** Ata Bulut
- **Student ID:** 220901582
- **GitHub Username:** atablt07

---

## 1. Overview of Contributions
In this project, I was responsible for developing the Hidden Markov Model (HMM), which serves as the baseline sequence labeling model. My primary deliverables included estimating transition and emission probabilities from training data and implementing a robust Viterbi decoding algorithm that strictly enforces BIO sequence constraints.

---

## 2. Specific Modules & Code Implemented
Detailed breakdown of components designed, written, or debugged by me:

### B. Model Architectures & Training (`src/models/hmm.py`)
- **HMM Baseline Implementation:** Developed the core `HiddenMarkovModel` class and the `fit` function using Maximum Likelihood Estimation (MLE) for transitions and emissions.
- **BIO Sequence Constraints:** Designed a custom transition matrix generator (`get_transition_matrix`) that strictly enforces BIO tagging rules (e.g., preventing `O` → `I-X` and `B-X` → `I-Y` transitions).
- **Probability Smoothing:** Applied unigram-prior smoothing with `alpha` and `beta` parameters to handle Out-Of-Vocabulary (OOV) words and balance emission probabilities.
- **Viterbi Decoding Algorithm:** Developed a log-probability based Viterbi decoder with path backtracking (`predict_sentence`) to predict valid BIO tags for newly tokenized sentences.

*(Note: Data preprocessing, BiLSTM-CRF, and CRF modules were developed by other team members and are excluded from this report.)*

---

## 3. Key Pull Requests / Git Commits
List of key commits / PRs contributed to the repository:
- `commit [hash]`: Added Hidden Markov Model baseline structure and MLE parameter estimation
- `commit [hash]`: Integrated strict BIO sequence constraints into the HMM transition matrix
- `commit [hash]`: Added unigram-prior smoothing for reliable emission probabilities
- `commit [hash]`: Implemented Viterbi decoding algorithm with path backtracking