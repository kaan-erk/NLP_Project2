"""
Data Preprocessing, Token Alignment, and Balanced Multi-Class Resampling Pipeline.
Ensures equal sample representation across all fine-grained toxic categories:
INSULT, IDENTITY_ATTACK, OBSCENE, THREAT, and OTHER_TOXIC.
"""

import ast
import json
import logging
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import spacy
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

# Supported Toxicity Categories in Order of Specificity / Inverse Frequency
PRIORITY_CATEGORIES = ["THREAT", "IDENTITY_ATTACK", "OBSCENE", "OTHER_TOXIC", "INSULT"]

def determine_specific_category(type_val: Any) -> str:
    """
    Extracts the most specific/minority category present in the annotations.
    Prevents the broad generic 'insult' umbrella from overshadowing 'threat' or 'identity_attack'.
    """
    if not type_val:
        return "OTHER_TOXIC"
    try:
        if isinstance(type_val, str):
            type_dict = ast.literal_eval(type_val)
        elif isinstance(type_val, dict):
            type_dict = type_val
        else:
            return "OTHER_TOXIC"

        if not type_dict:
            return "OTHER_TOXIC"

        norm_keys = {}
        for k, v in type_dict.items():
            k_clean = str(k).strip().lower()
            if "threat" in k_clean:
                norm_keys["THREAT"] = v
            elif "identity" in k_clean:
                norm_keys["IDENTITY_ATTACK"] = v
            elif "profane" in k_clean or "obscene" in k_clean:
                norm_keys["OBSCENE"] = v
            elif "other" in k_clean:
                norm_keys["OTHER_TOXIC"] = v
            elif "insult" in k_clean or "insu" in k_clean:
                norm_keys["INSULT"] = v

        for cat in PRIORITY_CATEGORIES:
            if cat in norm_keys and norm_keys[cat] > 0:
                return cat

        return "OTHER_TOXIC"
    except Exception:
        return "OTHER_TOXIC"

def clean_text(text: str) -> str:
    """Clean corrupted unicode and control characters while preserving lengths."""
    text = unicodedata.normalize("NFKC", text)
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

THREAT_WORDS = {
    "kill", "killing", "killed", "killer", "killers", "die", "death", "murder", "murdered", "murdering",
    "shoot", "shooting", "shot", "hang", "hanged", "beat", "beaten", "punch", "destroy",
    "destroying", "destroyed", "stab", "stabbed", "burn", "burned", "burning", "rape", "raped",
    "raping", "bomb", "bombing", "bombed", "execute", "executed", "choke", "choking", "slit",
    "throat", "bullet", "assassinate", "dead"
}

OBSCENE_WORDS = {
    "fuck", "fucking", "fucked", "fucker", "fuckers", "shit", "shitty", "bullshit", "bitch",
    "bitches", "bitching", "ass", "asshole", "assholes", "cunt", "cunts", "dick", "dicks",
    "pussy", "bastard", "bastards", "damn", "crap", "d*ckhead", "dickhead", "cock", "tits",
    "prick", "whore", "slut"
}

IDENTITY_WORDS = {
    "racist", "racists", "racism", "bigot", "bigots", "bigotry", "gay", "queer", "fag", "faggot", "faggots",
    "dyke", "nigger", "nigga", "niggers", "negro", "retard", "retarded", "spic", "chink", "kike",
    "tranny", "transgender", "muslims", "islamist", "jews", "jewish", "blacks", "whites",
    "immigrants", "mexicans", "homosexual"
}

INSULT_WORDS = {
    "idiot", "idiots", "idiotic", "moron", "morons", "stupid", "stupidity", "fool", "fools",
    "foolish", "clown", "clowns", "loser", "losers", "scumbag", "scumbags", "dumb", "dumber",
    "dumbest", "jerk", "jerks", "buffoon", "buffoons", "ignorant", "ignorance", "disgusting",
    "pathetic", "ugly", "hypocrite", "hypocrites", "liar", "liars", "cheat", "corrupt", "trash",
    "garbage", "pig", "pigs", "filth", "filthy", "incompetent", "imbecile", "useless", "coward",
    "cowards", "shame", "scum"
}

def get_token_aspect(token_lower: str, doc_category: str) -> str:
    """Disambiguate true token-level semantic category based on linguistic lexicon."""
    if token_lower in THREAT_WORDS:
        return "THREAT"
    if token_lower in OBSCENE_WORDS:
        return "OBSCENE"
    if token_lower in IDENTITY_WORDS:
        return "IDENTITY_ATTACK"
    if token_lower in INSULT_WORDS:
        return "INSULT"
    return doc_category

def align_doc_to_bio(doc, positions: set, category: str) -> List[str]:
    """
    Align token character spans with toxic character offsets to construct strict BIO tags
    using token-level aspect disambiguation.
    """
    bio_tags = []
    in_span = False
    curr_cat = category
    
    for token in doc:
        token_len = len(token.text)
        if token_len == 0 or token.text.isspace():
            bio_tags.append("O")
            in_span = False
            continue
            
        token_chars = set(range(token.idx, token.idx + token_len))
        overlap = token_chars.intersection(positions)
        
        is_token_toxic = (len(overlap) >= token_len * 0.5) if token_len > 1 else (len(overlap) > 0)
        
        if is_token_toxic and positions and category != "NEUTRAL_O":
            tok_cat = get_token_aspect(token.text.lower(), category)
            if not in_span:
                curr_cat = tok_cat
                bio_tags.append(f"B-{curr_cat}")
                in_span = True
            else:
                if tok_cat != curr_cat and tok_cat in {"THREAT", "OBSCENE", "IDENTITY_ATTACK", "INSULT"}:
                    curr_cat = tok_cat
                    bio_tags.append(f"B-{curr_cat}")
                else:
                    bio_tags.append(f"I-{curr_cat}")
        else:
            bio_tags.append("O")
            in_span = False
            
    return bio_tags

def load_and_group_raw(file_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Loads raw jsonl and groups samples by fine-grained category."""
    grouped = defaultdict(list)
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                pos = parse_positions(item.get("position"))
                if pos:
                    cat = determine_specific_category(item.get("type"))
                    grouped[cat].append(item)
                else:
                    grouped["NEUTRAL_O"].append(item)
    return grouped

def create_balanced_split(
    grouped_data: Dict[str, List[Dict[str, Any]]],
    target_per_class: int,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Extracts an exactly equal number of samples per toxic category,
    plus the same target number of neutral sentences.
    """
    random.seed(seed)
    balanced_samples = []

    for cat in PRIORITY_CATEGORIES:
        pool = grouped_data.get(cat, [])
        if len(pool) >= target_per_class:
            selected = random.sample(pool, target_per_class)
        else:
            # Oversample minority class to reach exact target
            selected = pool + random.choices(pool, k=target_per_class - len(pool))
            
        for item in selected:
            sample_copy = dict(item)
            sample_copy["assigned_category"] = cat
            balanced_samples.append(sample_copy)

    # Add neutral sentences
    neutral_pool = grouped_data.get("NEUTRAL_O", [])
    if len(neutral_pool) >= target_per_class:
        selected_neutral = random.sample(neutral_pool, target_per_class)
    else:
        selected_neutral = neutral_pool + random.choices(neutral_pool, k=target_per_class - len(neutral_pool))
        
    for item in selected_neutral:
        sample_copy = dict(item)
        sample_copy["assigned_category"] = "NEUTRAL_O"
        balanced_samples.append(sample_copy)

    random.shuffle(balanced_samples)
    return balanced_samples

def process_and_align_samples(raw_samples: List[Dict[str, Any]], nlp) -> List[Dict[str, Any]]:
    """Runs spaCy pipeline, token alignment, and extracts linguistic features."""
    texts = [clean_text(r.get("text_of_post", "")) for r in raw_samples]
    docs = list(nlp.pipe(texts, batch_size=256))
    
    processed = []
    for idx, (rec, doc) in enumerate(zip(raw_samples, docs)):
        positions = parse_positions(rec.get("position"))
        category = rec.get("assigned_category", determine_specific_category(rec.get("type")))
        
        if category == "NEUTRAL_O":
            positions = set()
            
        bio_tags = align_doc_to_bio(doc, positions, category)
        
        tokens = [t.text for t in doc]
        lemmas = [t.lemma_ for t in doc]
        pos_tags = [t.pos_ for t in doc]
        
        processed.append({
            "id": idx,
            "text": doc.text,
            "tokens": tokens,
            "lemmas": lemmas,
            "pos_tags": pos_tags,
            "tags": bio_tags,
            "category": category,
            "is_toxic": int(category != "NEUTRAL_O" and bool(positions))
        })
        
    return processed

def build_tag_vocabulary() -> Dict[str, int]:
    """Canonical 11 BIO tags + special sequence model tokens."""
    base_tags = ["O"]
    for cat in PRIORITY_CATEGORIES:
        base_tags.extend([f"B-{cat}", f"I-{cat}"])
        
    all_tags = sorted(list(set(base_tags)))
    tag_to_ix = {tag: idx for idx, tag in enumerate(all_tags)}
    
    special_tokens = ["<START>", "<STOP>", "<PAD>"]
    for st in special_tokens:
        if st not in tag_to_ix:
            tag_to_ix[st] = len(tag_to_ix)
            
    return tag_to_ix

def build_word_vocabulary(samples: List[Dict[str, Any]], min_freq: int = 2) -> Dict[str, int]:
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
    
    grouped_train = load_and_group_raw(raw_train_file)
    grouped_test = load_and_group_raw(raw_test_file)
    
    print("\n--- RAW POOL COUNTS ---")
    print("Train raw available:")
    for k, v in grouped_train.items():
        print(f"  {k:<20}: {len(v)}")
    print("Test raw available:")
    for k, v in grouped_test.items():
        print(f"  {k:<20}: {len(v)}")
        
    # Balanced Sampling:
    # Train: 1400 per class (1400 * 5 toxic = 7,000 toxic + 1400 neutral = 8,400)
    # Val: 150 per class (150 * 5 toxic = 750 toxic + 150 neutral = 900)
    # Test: 80 per class (80 * 5 toxic = 400 toxic + 80 neutral = 480)
    train_balanced_raw = create_balanced_split(grouped_train, target_per_class=1400, seed=42)
    val_balanced_raw = create_balanced_split(grouped_train, target_per_class=150, seed=123)
    test_balanced_raw = create_balanced_split(grouped_test, target_per_class=80, seed=999)
    
    logging.info(f"Processing and aligning {len(train_balanced_raw)} train samples...")
    train_samples = process_and_align_samples(train_balanced_raw, nlp)
    
    logging.info(f"Processing and aligning {len(val_balanced_raw)} val samples...")
    val_samples = process_and_align_samples(val_balanced_raw, nlp)
    
    logging.info(f"Processing and aligning {len(test_balanced_raw)} test samples...")
    test_samples = process_and_align_samples(test_balanced_raw, nlp)
    
    # Save splits
    for name, data in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        out_path = PROCESSED_DATA_DIR / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logging.info(f"Saved {len(data)} examples to {out_path}")
        
    tag_to_ix = build_tag_vocabulary()
    word_to_ix = build_word_vocabulary(train_samples, min_freq=2)
    
    with open(PROCESSED_DATA_DIR / "tag_to_ix.json", "w", encoding="utf-8") as f:
        json.dump(tag_to_ix, f, indent=2, ensure_ascii=False)
        
    with open(PROCESSED_DATA_DIR / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(word_to_ix, f, indent=2, ensure_ascii=False)
        
    # Compute Statistics
    train_tag_counts = Counter()
    test_tag_counts = Counter()
    
    for s in train_samples:
        train_tag_counts.update(s["tags"])
    for s in test_samples:
        test_tag_counts.update(s["tags"])
        
    stats = {
        "dataset_name": "Multi-Aspect Toxic Spans (Balanced Resampling)",
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "vocabulary_size": len(word_to_ix),
        "total_tags": len(tag_to_ix),
        "tag_to_ix": tag_to_ix,
        "train_tag_distribution": dict(train_tag_counts.most_common()),
        "test_tag_distribution": dict(test_tag_counts.most_common())
    }
    
    with open(PROCESSED_DATA_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
        
    print("\n--- BALANCED TRAIN TAG DISTRIBUTION ---")
    for tag, cnt in train_tag_counts.most_common():
        print(f"  {tag:<22}: {cnt}")
        
    print("\n--- BALANCED TEST TAG DISTRIBUTION ---")
    for tag, cnt in test_tag_counts.most_common():
        print(f"  {tag:<22}: {cnt}")

if __name__ == "__main__":
    main()
