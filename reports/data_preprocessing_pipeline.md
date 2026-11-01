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

### Step 4: Inverse-Frequency Multi-Label Disambiguation
In raw social media comments, almost every severe comment labeled as `threat` or `identity_attack` is also co-labeled as `insult`. A naive majority-vote resolution causes `insult` to dominate over 85% of all toxic spans, completely starving minority categories.
To guarantee semantic diversity, we enforce an **Inverse-Frequency Priority Rule**:
$$\text{Priority: } \text{THREAT} > \text{IDENTITY\_ATTACK} > \text{OBSCENE} > \text{OTHER\_TOXIC} > \text{INSULT}$$

### Step 5: Stratified Balanced Resampling
To eliminate extreme class imbalance, we implemented exact stratified resampling across all 5 toxic categories, pairing them with an equal portion of clean neutral comments:
- **Train Split:** Exactly 400 sentences per toxic category (2,000 toxic sentences) + 400 clean neutral sentences = 2,400 sentences total.
- **Validation Split:** Exactly 50 sentences per category = 300 sentences total.
- **Test Split:** Exactly 40 sentences per category = 240 sentences total.

---

## 4. Balanced Dataset Statistics

### Dataset Splits
- **Training Set (`train.jsonl`):** 2,400 samples
- **Validation Set (`val.jsonl`):** 300 samples
- **Test Set (`test.jsonl`):** 240 samples
- **Vocabulary Size:** 4,717 unique words ($N_{\min}=2$).

### Balanced Tag Distribution

| Tag | Train Token Count | Test Token Count | Ratio Balance |
| :--- | :--- | :--- | :--- |
| **`B-OBSCENE`** | 473 | 48 | Balanced (~20%) |
| **`B-IDENTITY_ATTACK`** | 473 | 45 | Balanced (~20%) |
| **`B-INSULT`** | 455 | 44 | Balanced (~20%) |
| **`B-THREAT`** | 453 | 45 | Balanced (~20%) |
| **`B-OTHER_TOXIC`** | 449 | 46 | Balanced (~20%) |
| `I-THREAT` | 235 | 10 | Context-dependent |
| `I-IDENTITY_ATTACK`| 141 | 15 | Context-dependent |
| `I-OBSCENE` | 84 | 6 | Context-dependent |
| `I-OTHER_TOXIC` | 65 | 6 | Context-dependent |
| `I-INSULT` | 28 | 3 | Context-dependent |
| `O` (Neutral Tokens)| 95,507 | 9,305 | Natural sequence surplus |

> [!NOTE]
> In natural sequence labeling tasks, `O` represents the surrounding non-toxic sentence syntax (grammatical background) and will naturally exceed entity tokens in quantity. However, the 5 toxic entity categories now possess strictly equal, balanced representations (~450-473 `B-` tokens each in training, and ~44-48 `B-` tokens each in testing), completely eliminating minority starvation.

---

## 5. Execution Pipeline
Run the data pipeline end-to-end:
```bash
python src/data/download.py
python src/data/preprocess.py
```
Preprocessed data splits and summary metadata are written to `data/processed/`.
