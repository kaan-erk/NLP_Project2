"""
Data Preprocessing and Token Alignment Pipeline.
Converts raw character offsets from SemEval Toxic Spans into token-level BIO tags.
Enriches tokens with POS tags and lemmas using spaCy.
Produces train/val/test splits, vocabulary, and tagsets.
"""

import ast
import json
import logging
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any

import spacy
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

# Supported Toxicity Categories
VALID_CATEGORIES = ["INSULT", "IDENTITY_ATTACK", "OBSCENE", "THREAT", "OTHER_TOXIC"]

def normalize_category(cat_str: str) -> str:
    """Normalize raw category names into standard class keys."""
    cat = cat_str.strip().lower()
    if cat in ["insult", "insu"]:
        return "INSULT"
    elif "identity" in cat:
        return "IDENTITY_ATTACK"
    elif "profane" in cat or "obscene" in cat:
        return "OBSCENE"
    elif "threat" in cat:
        return "THREAT"
    else:
        return "OTHER_TOXIC"

def determine_dominant_category(type_str: Any) -> str:
    """Extract dominant category from raw type dictionary string."""
    if not type_str:
        return "OTHER_TOXIC"
    try:
        if isinstance(type_str, str):
            type_dict = ast.literal_eval(type_str)
        elif isinstance(type_str, dict):
            type_dict = type_str
        else:
            return "OTHER_TOXIC"

        if not type_dict:
            return "OTHER_TOXIC"
        
        # Sort by confidence score descending
        sorted_types = sorted(type_dict.items(), key=lambda x: x[1], reverse=True)
        return normalize_category(sorted_types[0][0])
    except Exception:
        return "OTHER_TOXIC"

def clean_text(text: str) -> str:
    """Clean corrupted unicode and control characters while preserving lengths."""
    text = unicodedata.normalize("NFKC", text)
    # Replace non-printable characters except standard whitespace
    cleaned = "".join(c if (c.isprintable() or c in "\n\r\t") else " " for c in text)
    return cleaned

def parse_positions(pos_val: Any) -> set:
    """Parse character position indices from string or list."""
    if not pos_val:
        return set()
    try:
        if isinstance(pos_val, str):
            pos_list = ast.literal_eval(pos_val)
        elif isinstance(pos_val, (list, set)):
            pos_list = pos_val
        else:
            return set()
        return set(int(p) for p in pos_list)
    except Exception:
        return set()

def align_doc_to_bio(doc, positions: set, category: str) -> List[str]:
    """
    Align token character spans with toxic character offsets to construct BIO tags.
    """
    bio_tags = []
    in_span = False
    
    for token in doc:
        # Ignore whitespace tokens if spaCy produces any
        token_len = len(token.text)
        if token_len == 0 or token.text.isspace():
            bio_tags.append("O")
            in_span = False
            continue
            
        token_chars = set(range(token.idx, token.idx + token_len))
        overlap = token_chars.intersection(positions)
        
        # Token is toxic if at least 50% of characters or >= 1 char for single-char tokens
        is_token_toxic = (len(overlap) >= token_len * 0.5) if token_len > 1 else (len(overlap) > 0)
        
        if is_token_toxic and positions:
            if not in_span:
                bio_tags.append(f"B-{category}")
                in_span = True
            else:
                bio_tags.append(f"I-{category}")
        else:
            bio_tags.append("O")
            in_span = False
            
    return bio_tags

def process_dataset_split(raw_file: Path, nlp) -> List[Dict[str, Any]]:
    """Process a raw jsonl file and generate tokenized BIO sequences."""
    records = []
    with open(raw_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
                
    logging.info(f"Processing {len(records)} records from {raw_file.name}...")
    
    texts = [clean_text(r.get("text_of_post", "")) for r in records]
    docs = list(nlp.pipe(texts, batch_size=256))
    
    processed_samples = []
    for idx, (rec, doc) in enumerate(zip(records, docs)):
        positions = parse_positions(rec.get("position"))
        category = determine_dominant_category(rec.get("type"))
        is_toxic = int(rec.get("toxic", 1 if positions else 0))
        
        # If record is marked non-toxic, positions are empty
        if is_toxic == 0:
            positions = set()
            
        bio_tags = align_doc_to_bio(doc, positions, category)
        
        tokens = [t.text for t in doc]
        lemmas = [t.lemma_ for t in doc]
        pos_tags = [t.pos_ for t in doc]
        
        processed_samples.append({
            "id": idx,
            "text": doc.text,
            "tokens": tokens,
            "lemmas": lemmas,
            "pos_tags": pos_tags,
            "tags": bio_tags,
            "category": category if is_toxic and positions else "NONE",
            "is_toxic": is_toxic
        })
        
    return processed_samples

def build_tag_vocabulary(samples: List[Dict[str, Any]]) -> Dict[str, int]:
    """Build canonical BIO tag mapping with special tokens."""
    tags = set()
    for s in samples:
        tags.update(s["tags"])
        
    # Ensure all 11 core classes are present
    base_tags = ["O"]
    for cat in VALID_CATEGORIES:
        base_tags.extend([f"B-{cat}", f"I-{cat}"])
        
    all_tags = sorted(list(tags.union(base_tags)))
    tag_to_ix = {tag: idx for idx, tag in enumerate(all_tags)}
    
    # Add special tokens for CRF
    special_tokens = ["<START>", "<STOP>", "<PAD>"]
    for st in special_tokens:
        if st not in tag_to_ix:
            tag_to_ix[st] = len(tag_to_ix)
            
    return tag_to_ix

def build_word_vocabulary(samples: List[Dict[str, Any]], min_freq: int = 2) -> Dict[str, int]:
    """Build word-to-index mapping based on frequency."""
    counter = Counter()
    for s in samples:
        counter.update([tok.lower() for tok in s["tokens"]])
        
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, count in counter.most_common():
        if count >= min_freq:
            vocab[word] = len(vocab)
            
    return vocab

def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("Loading spaCy 'en_core_web_sm' model...")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    nlp.enable_pipe("tagger")
    
    raw_train_file = RAW_DATA_DIR / "toxic_spans_train_raw.jsonl"
    raw_test_file = RAW_DATA_DIR / "toxic_spans_test_raw.jsonl"
    
    if not raw_train_file.exists() or not raw_test_file.exists():
        raise FileNotFoundError("Raw data files not found. Run download.py first!")
        
    all_train_samples = process_dataset_split(raw_train_file, nlp)
    test_samples = process_dataset_split(raw_test_file, nlp)
    
    # Split train into train (85%) and validation (15%) with fixed seed
    import random
    random.seed(42)
    random.shuffle(all_train_samples)
    
    val_size = int(len(all_train_samples) * 0.15)
    val_samples = all_train_samples[:val_size]
    train_samples = all_train_samples[val_size:]
    
    logging.info(f"Split sizes -> Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    # Save splits
    for name, data in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        out_path = PROCESSED_DATA_DIR / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logging.info(f"Saved {len(data)} examples to {out_path}")
        
    # Build Vocabularies
    tag_to_ix = build_tag_vocabulary(train_samples + val_samples + test_samples)
    word_to_ix = build_word_vocabulary(train_samples, min_freq=2)
    
    with open(PROCESSED_DATA_DIR / "tag_to_ix.json", "w", encoding="utf-8") as f:
        json.dump(tag_to_ix, f, indent=2, ensure_ascii=False)
        
    with open(PROCESSED_DATA_DIR / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(word_to_ix, f, indent=2, ensure_ascii=False)
        
    # Compute Statistics
    tag_counter = Counter()
    total_tokens = 0
    toxic_span_count = 0
    
    for s in train_samples:
        total_tokens += len(s["tokens"])
        tag_counter.update(s["tags"])
        toxic_span_count += sum(1 for t in s["tags"] if t.startswith("B-"))
        
    stats = {
        "dataset_name": "Multi-Aspect Toxic Rationale Spans",
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "total_tokens_train": total_tokens,
        "vocabulary_size": len(word_to_ix),
        "total_tags": len(tag_to_ix),
        "tag_to_ix": tag_to_ix,
        "toxic_spans_in_train": toxic_span_count,
        "tag_distribution_train": dict(tag_counter.most_common())
    }
    
    with open(PROCESSED_DATA_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
        
    logging.info("Preprocessing complete! Dataset summary saved.")
    print("\n--- DATASET SUMMARY ---")
    print(f"Total Unique Tags: {len(tag_to_ix)} (including special tokens)")
    print(f"Vocabulary Size: {len(word_to_ix)} words")
    print("Tag Distribution (Train):")
    for tag, cnt in tag_counter.most_common():
        print(f"  {tag:<20}: {cnt}")

if __name__ == "__main__":
    main()

