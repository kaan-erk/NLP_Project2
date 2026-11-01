# Contribution Report: <STUDENT_NAME>

- **Student Name:** <STUDENT_NAME>
- **Student ID:** <STUDENT_ID>
- **GitHub Username:** <GITHUB_USERNAME>
- **File Naming Format:** `contributions/<STUDENT_NAME>_<STUDENT_ID>_<GITHUB_USERNAME>.md`  
  *(Example: `contributions/HASAN_KAYA_230123123_hasankayyya.md`)*

---

## 1. Overview of Contributions
Brief 2-3 sentence summary of the member's responsibilities and major deliverables in the project.

---

## 2. Specific Modules & Code Implemented
Detailed breakdown of components designed, written, or debugged by this member:

### A. Data Engineering & Preprocessing Pipeline
- Implemented `src/data/download.py` / `src/data/preprocess.py`.
- Designed character offset to token BIO alignment algorithm (`align_doc_to_bio`).
- Handled multi-class toxicity categorization and text sanitization.

### B. Model Architectures & Training
- Implemented sequence labeling baseline: HMM transition and emission estimation + Viterbi decoding in `src/models/hmm.py`.
- Extracted and engineered orthographic and context window features for Linear-Chain CRF in `src/models/crf.py`.
- Developed PyTorch BiLSTM-CRF forward log-sum-exp algorithm and Viterbi backtracking in `src/models/bilstm_crf.py`.

### C. Evaluation & Benchmarking
- Set up strict span-level F1 evaluation using `seqeval` in `src/utils/metrics.py`.
- Generated and analyzed learned transition matrix heatmaps.

---

## 3. Key Pull Requests / Git Commits
List of key commits / PRs contributed to the repository:
- `commit <hash>`: Added data preprocessing pipeline and spaCy token alignment
- `commit <hash>`: Implemented BiLSTM-CRF forward loss and Viterbi decoding
- `commit <hash>`: Added evaluation metrics and transition matrix visualization

