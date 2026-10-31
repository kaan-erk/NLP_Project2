# Individual Contribution Report

- **Student Name:** Kaan Erk Toraman
- **Student ID:** 220901564
- **Email:** 220901564@stu.istinye.edu.tr
- **GitHub Username:** kaan-erk
- **Repository Role:** Lead Data Engineer & Preprocessing Pipeline Architect

---

## 1. Overview of Contributions

In this project, I was primarily responsible for the **end-to-end Data Engineering, Preprocessing, and Token Alignment Pipeline**. 

Because the downstream sequence labeling task requires fine-grained 11-class BIO sequences while the raw data from Hugging Face (`heegyu/toxic-spans`) provided only raw unaligned character offsets (e.g., `[8, 9, 10, 11]`) and messy web text, I designed and implemented the entire pipeline from scratch to transform raw data into model-ready JSONL files, vocabularies, and PyTorch dataset loaders.

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

### D. Fine-Grained 11-Class BIO Taxonomy Construction
- Normalized raw category annotations into 5 core toxicity dimensions + `O`:
  - `B-INSULT`, `I-INSULT`
  - `B-IDENTITY_ATTACK`, `I-IDENTITY_ATTACK`
  - `B-OBSCENE`, `I-OBSCENE`
  - `B-THREAT`, `I-THREAT`
  - `B-OTHER_TOXIC`, `I-OTHER_TOXIC`
  - `O`
- Added special transition symbols for probabilistic models: `<START>`, `<STOP>`, `<PAD>`.

### E. Split Generation, Vocabulary & Statistics
- Partitioned raw training data into an 85% train split (`data/processed/train.jsonl` - 8,506 samples) and a 15% validation split (`data/processed/val.jsonl` - 1,500 samples), preserving the 1,000-sample test benchmark (`data/processed/test.jsonl`).
- Built frequency-filtered word vocabulary (`data/processed/vocab.json`, 10,651 tokens, $N_{\min}=2$) with `<PAD>` and `<UNK>` tokens.
- Generated `data/processed/tag_to_ix.json` and computed statistical distributions (`data/processed/dataset_summary.json`).

### F. PyTorch Data Pipeline (`src/data/dataset.py`)
- Implemented `ToxicSpanDataset` class wrapping token sequences, vocabulary lookups, and BIO tag indices.
- Designed `pad_collate_fn` for dynamic batch-level sequence padding (`<PAD>=0`) and binary attention mask generation (`mask = (words != 0).float()`) required by the BiLSTM-CRF model.

### G. Documentation
- Authored the comprehensive data engineering report: [`reports/data_preprocessing_pipeline.md`](../reports/data_preprocessing_pipeline.md).

---

## 3. Key Commits & GitHub Diffs
My contributions are reflected in the following core commits:
- Data acquisition script and raw data storage (`src/data/download.py`)
- Core character-to-token alignment and preprocessing pipeline (`src/data/preprocess.py`)
- PyTorch dataset and dynamic collate functions (`src/data/dataset.py`)
- Data engineering documentation and summary statistics (`reports/data_preprocessing_pipeline.md`, `data/processed/`)
