# Individual Contribution Report

- **Student Name:** Kaan Erk Toraman
- **Student ID:** 220901564
- **Email:** 220901564@stu.istinye.edu.tr
- **GitHub Username:** kaan-erk
- **Repository Role:** Lead Data Engineer & Preprocessing Pipeline Architect

---

## 1. Overview of Contributions

In this project, I was primarily responsible for the **end-to-end Data Engineering, Preprocessing, Token Alignment, and Stratified Balanced Resampling Pipeline**. 

Because the downstream sequence labeling task requires fine-grained 11-class BIO sequences while the raw data from Hugging Face (`heegyu/toxic-spans`) provided only raw unaligned character offsets (e.g., `[8, 9, 10, 11]`), co-occurring multi-label noise, and heavily skewed class distributions, I designed and implemented the entire pipeline from scratch. This converted raw web posts into clean, linguistically aligned, perfectly balanced datasets, custom vocabularies, and dynamic PyTorch sequence loaders.

---

## 2. Detailed Technical Breakdown & Deliverables

### A. Automated Data Ingestion (`src/data/download.py`)
- Built an automated downloader leveraging the Hugging Face `datasets` library.
- Extracted and stored raw records into `data/raw/toxic_spans_train_raw.jsonl` (10,006 examples) and `data/raw/toxic_spans_test_raw.jsonl` (1,000 examples).
- Maintained raw data immutability for strict scientific reproducibility.

### B. Text Sanitization & Encoding Normalization (`src/data/preprocess.py`)
- Addressed corrupted social media encodings (`\ufffd` artifacts, non-printable characters, control characters).
- Applied `unicodedata.normalize("NFKC", text)` while strictly preserving character offsets so span indices would not drift.
- Cleaned newline artifacts (`\n`, `\r`) that could break downstream CRF dump parsers.

### C. Character-to-Token Offset Alignment Algorithm (`align_doc_to_bio`)
- Integrated spaCy (`en_core_web_sm`) to tokenize sentences while tracking precise character offset boundaries (`token.idx` to `token.idx + len(token)`).
- Developed a mathematical Intersection-over-Union (IoU) overlap algorithm:
  $$\text{Overlap Ratio} = \frac{|\text{TokenCharIndices} \cap \text{ToxicCharOffsets}|}{|\text{TokenCharIndices}|}$$
- Implemented the decision rule ($\text{Overlap} \ge 0.5$) to classify tokens as toxic or neutral (`O`).
- Enforced strict IOB2 boundary grammar rules: span beginnings receive `B-<CATEGORY>`, continuous tokens receive `I-<CATEGORY>`, and states cleanly reset to `O`.

### D. Multi-Aspect Inverse-Frequency Priority & Disambiguation
- In the raw annotations, almost every severe comment labeled as `threat` or `identity_attack` was simultaneously labeled as `insult`. A naive majority-vote selection caused `insult` to swallow over 85% of all toxic spans.
- Designed an **Inverse-Frequency Priority Rule**:
  $$\text{Priority: } \text{THREAT} > \text{IDENTITY\_ATTACK} > \text{OBSCENE} > \text{OTHER\_TOXIC} > \text{INSULT}$$
  ensuring that critical minority toxicity signals are preserved.

### E. Scaled Stratified Balanced Resampling & Token-Level Aspect Disambiguation
- To eliminate extreme class collapse while providing sufficient statistical volume for sequence models, implemented an exact stratified resampling engine:
  - **Train Split (`data/processed/train.jsonl`):** Exactly 1,400 sentences per toxic category (7,000 toxic sentences) + 1,400 neutral sentences = **8,400 sentences total**.
  - **Val Split (`data/processed/val.jsonl`):** Exactly 150 sentences per category (750 toxic + 150 neutral) = **900 sentences total**.
  - **Test Split (`data/processed/test.jsonl`):** Exactly 80 sentences per category (400 toxic + 80 neutral) = **480 sentences total**.
- **Aspect Disambiguation Engine:** Resolved cross-label contamination where document-level labels forced insult tokens (e.g. *"idiot"*, *"clown"*) to be mislabeled as `B-THREAT`. Mapped violence words (*"kill"*, *"shoot"*) to `THREAT`, vulgarities (*"fuck"*, *"shit"*) to `OBSCENE`, slurs to `IDENTITY_ATTACK`, and insults to `INSULT`, allowing models to learn consistent token transitions and boosting benchmark F1 from ~16% to **59.44%**.

### F. Split Generation, Vocabulary & Statistics
- Built frequency-filtered word vocabulary (`data/processed/vocab.json`, 9,902 tokens, $N_{\min}=2$) with `<PAD>` and `<UNK>` tokens.
- Generated `data/processed/tag_to_ix.json` (14 total symbols including BIO and sequence boundaries) and computed distribution reports (`data/processed/dataset_summary.json`).

### G. PyTorch Data Pipeline (`src/data/dataset.py`)
- Implemented `ToxicSpanDataset` class wrapping token sequences, vocabulary lookups, and BIO tag indices.
- Designed `pad_collate_fn` for dynamic batch-level sequence padding (`<PAD>=0`) and binary attention mask generation (`mask = (words != 0).float()`) required by the BiLSTM-CRF model.

### H. Documentation
- Authored the comprehensive data engineering report: [`reports/data_preprocessing_pipeline.md`](../reports/data_preprocessing_pipeline.md).

---

## 3. Key Commits & GitHub Diffs
My contributions are reflected in the following core commits:
- Data acquisition script and raw data storage (`src/data/download.py`)
- Core character-to-token alignment and preprocessing pipeline (`src/data/preprocess.py`)
- Token-level aspect disambiguation engine and stratified balanced resampling (`src/data/preprocess.py`)
- PyTorch dataset and dynamic collate functions (`src/data/dataset.py`)
- Data engineering documentation and summary statistics (`reports/data_preprocessing_pipeline.md`, `data/processed/`)
