# Data Preprocessing & Sequence Alignment Pipeline Documentation

## 1. Project Overview & Task Formulation
In this project, we address **Multi-Aspect Token-Level Rationale & Toxic Span Extraction** formulated as a sequence labeling (token classification) problem under Explainable AI (XAI).

Unlike conventional binary toxic classification (which assigns a single label to an entire text) or binary span detection, our task identifies the **exact token boundaries** and assigns **fine-grained semantic categories** of abusive language using an 11-class **BIO (Beginning, Inside, Outside)** scheme:

- `O`: Neutral / non-toxic tokens.
- `B-INSULT`, `I-INSULT`: Insults, name-calling, and disparaging language.
- `B-IDENTITY_ATTACK`, `I-IDENTITY_ATTACK`: Slurs and hatred targeted at identity, race, or religion.
- `B-OBSCENE`, `I-OBSCENE`: Profanities, vulgarities, and obscene expressions.
- `B-THREAT`, `I-THREAT`: Physical violence, harm, or intimidatory threats.
- `B-OTHER_TOXIC`, `I-OTHER_TOXIC`: Ambiguous or generalized toxic statements.

Special transition tokens for probabilistic sequence models: `<START>`, `<STOP>`, and `<PAD>`.

---

## 2. Raw Dataset Details
- **Primary Source:** `heegyu/toxic-spans` (SemEval-2021 Task 5 dataset with fine-grained toxicity attributes from Civil Comments).
- **Raw Volume:** 10,006 training examples and 1,000 test examples.
- **Raw Data Format:**
  - `text_of_post`: Unprocessed comment text.
  - `position`: Stringified list of character-level offset integers (e.g., `"[8, 9, 10, 11, 12]"`).
  - `probability`: Span tuples mapped to annotator agreement scores.
  - `type`: Distribution over fine-grained toxicity categories (e.g., `"{'insult': 1.0}"`).
  - `toxic`: Binary toxicity indicator (0 or 1).

---

## 3. Data Engineering & Alignment Methodology

### Step 1: Text Sanitization & Encoding Cleanup
Raw web and social media comments contain corrupted byte sequences (e.g., `\ufffd`), control characters, and Unicode discrepancies.
We apply:
```python
text = unicodedata.normalize("NFKC", text)
cleaned = "".join(c if (c.isprintable() or c in "\n\r\t") else " " for c in text)
```

### Step 2: Linguistic Tokenization & Feature Enrichment
We tokenize sentences using spaCy (`en_core_web_sm`). For every token, spaCy provides:
- Word text: `token.text`
- Absolute start character index: `token.idx`
- Token character length: `len(token.text)`
- Lemma: `token.lemma_`
- Part-of-Speech tag: `token.pos_`

### Step 3: Character-to-Token Offset Alignment Algorithm
Because human annotators mark character spans while sequence models operate on discrete token sequences, we map character indices to tokens using an overlap ratio threshold:
$$\text{Overlap Ratio} = \frac{|\text{TokenChars} \cap \text{ToxicCharOffsets}|}{|\text{TokenChars}|}$$
A token is marked toxic if $\text{Overlap Ratio} \ge 0.5$ (or $\ge 1$ character for single-character punctuation).

### Step 4: BIO Boundary Enforcement
1. When transitioning from `O` to a toxic token with category $C$, the token receives tag `B-C`.
2. Subsequent contiguous toxic tokens with the same category receive tag `I-C`.
3. If non-toxic tokens intervene, the state resets to `O`.

---

## 4. Dataset Splits & Statistics
- **Training Set (`train.jsonl`):** 8,506 samples (85% split)
- **Validation Set (`val.jsonl`):** 1,500 samples (15% split)
- **Test Set (`test.jsonl`):** 1,000 samples (Independent test split)
- **Vocabulary Size:** 10,651 unique words ($N_{\min}=2$).
- **Tag Distribution in Training:**
  - `O`: 370,749 tokens
  - `B-INSULT`: 4,285 tokens | `I-INSULT`: 522 tokens
  - `B-OBSCENE`: 305 tokens | `I-OBSCENE`: 69 tokens
  - `B-IDENTITY_ATTACK`: 184 tokens | `I-IDENTITY_ATTACK`: 139 tokens
  - `B-OTHER_TOXIC`: 90 tokens | `I-OTHER_TOXIC`: 31 tokens
  - `B-THREAT`: 64 tokens | `I-THREAT`: 78 tokens

---

## 5. Execution
Run the data pipeline end-to-end:
```bash
python src/data/download.py
python src/data/preprocess.py
```
Outputs are written to `data/processed/`.

