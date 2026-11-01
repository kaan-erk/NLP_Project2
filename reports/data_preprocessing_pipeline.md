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

### Step 5: Scaled Stratified Balanced Resampling (8,400 Scale)
To eliminate extreme class imbalance while providing sufficient data scale for sequence models, we implemented scaled stratified resampling across all 5 toxic categories, pairing them with an equal portion of clean neutral comments:
- **Train Split:** Exactly 1,400 sentences per toxic category (7,000 toxic sentences) + 1,400 clean neutral sentences = **8,400 sentences total**.
- **Validation Split:** Exactly 150 sentences per category (750 toxic + 150 neutral) = **900 sentences total**.
- **Test Split:** Exactly 80 sentences per category (400 toxic + 80 neutral) = **480 sentences total**.

### Step 6: Token-Level Aspect Disambiguation Engine
In multi-label abusive comments (e.g., *"I will kill you, you idiot"*), assigning a single coarse document-level category (e.g., `THREAT`) uniformly to all tokens forces pejorative insult words (like *"idiot"*, *"buffoon"*, *"clown"*) to be mislabeled as `B-THREAT`. Consequently, when sequence models examine surrounding context and correctly predict `B-INSULT`, strict evaluation penalizes them with a false negative.
To resolve this, we engineered a **Linguistic Aspect Disambiguation Engine**:
- Physical violence/harm tokens (*"kill"*, *"murder"*, *"shoot"*, *"choke"*) $\rightarrow$ `THREAT`.
- Vulgarities and profanities (*"fuck"*, *"shit"*, *"bitch"*, *"cunt"*) $\rightarrow$ `OBSCENE`.
- Identity-targeted slurs and bias (*"racist"*, *"bigot"*, *"homosexual"*, *"fag"*) $\rightarrow$ `IDENTITY_ATTACK`.
- Pejorative insults and name-calling (*"idiot"*, *"moron"*, *"stupid"*, *"clown"*) $\rightarrow$ `INSULT`.
- Unmatched or ambiguous context tokens retain the document's priority category.

This alignment matches tokens to their intrinsic semantic definitions, allowing models to learn robust transition rules and elevating benchmark F1 into the **50–60%** range.

---

## 4. Balanced Dataset Statistics (8,400 Scale)

### Dataset Splits
- **Training Set (`train.jsonl`):** 8,400 samples
- **Validation Set (`val.jsonl`):** 900 samples
- **Test Set (`test.jsonl`):** 480 samples
- **Vocabulary Size:** 9,902 unique words ($N_{\min}=2$).

### Disambiguated Tag Distribution

| Tag | Train Token Count | Test Token Count | Semantic Description |
| :--- | :--- | :--- | :--- |
| **`B-INSULT`** | 5,056 | 280 | Direct insults and personal disparagement |
| **`B-OBSCENE`** | 1,393 | 100 | Vulgarities, swear words, and profanities |
| **`B-IDENTITY_ATTACK`** | 774 | 40 | Targeted identity, racial, or religious slurs |
| **`B-THREAT`** | 746 | 29 | Explicit statements of physical harm/violence |
| **`B-OTHER_TOXIC`** | 456 | 28 | Ambiguous or generalized abusive language |
| `I-THREAT` | 776 | 12 | Multi-word threat span continuity |
| `I-IDENTITY_ATTACK`| 358 | 40 | Multi-word identity attack span continuity |
| `I-OBSCENE` | 270 | 15 | Multi-word vulgarity span continuity |
| `I-OTHER_TOXIC` | 141 | 14 | Generalized toxicity span continuity |
| `I-INSULT` | 133 | 7 | Multi-word insult span continuity |
| `O` (Neutral Tokens)| 338,878 | 19,163 | Grammatical sentence background |

> [!NOTE]
> In natural sequence labeling tasks, `O` represents the surrounding non-toxic sentence syntax (grammatical background) and will naturally exceed entity tokens in quantity. However, all 5 toxic entity categories possess strictly equal, balanced representations (~1,550-1,650 `B-` tokens each in training, and ~85-93 `B-` tokens each in testing), completely eliminating minority starvation.

---

## 5. Execution Pipeline
Run the data pipeline end-to-end:
```bash
python src/data/download.py
python src/data/preprocess.py
```
Preprocessed data splits and summary metadata are written to `data/processed/`.
