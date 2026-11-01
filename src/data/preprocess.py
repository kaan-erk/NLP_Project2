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

PRIORITY_CATEGORIES = ["THREAT", "IDENTITY_ATTACK", "OBSCENE", "OTHER_TOXIC", "INSULT"]

def determine_specific_category(type_val: Any) -> str:
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
    text = unicodedata.normalize("NFKC", text)
    cleaned = "".join(c if (c.isprintable() or c in "\n\r\t") else " " for c in text)
    return cleaned

def parse_positions(pos_val: Any) -> set:
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
    "shoot", "shooting", "shot", "hang", "hanged", "hanging", "beat", "beaten", "punch", "punched", "destroy",
    "destroying", "destroyed", "stab", "stabbed", "stabbing", "burn", "burned", "burning", "rape", "raped",
    "raping", "bomb", "bombing", "bombed", "execute", "executed", "execution", "choke", "choking", "slit",
    "throat", "bullet", "assassinate", "assassination", "dead", "torture", "tortured", "torturing", "slaughter",
    "butcher", "drown", "massacre", "poison", "assault", "strangle", "terminate", "attack", "hurt", "harm"
}

OBSCENE_WORDS = {
    "fuck", "fucking", "fucked", "fucker", "fuckers", "shit", "shitty", "bullshit", "bitch",
    "bitches", "bitching", "ass", "asshole", "assholes", "cunt", "cunts", "dick", "dicks",
    "pussy", "bastard", "bastards", "damn", "crap", "d*ckhead", "dickhead", "cock", "cocks", "tits",
    "prick", "whore", "whores", "slut", "sluts", "dipshit", "wanker", "jackass"
}

IDENTITY_WORDS = {
    "racist", "racists", "racism", "bigot", "bigots", "bigotry", "gay", "queer", "fag", "faggot", "faggots",
    "dyke", "nigger", "nigga", "niggers", "negro", "retard", "retarded", "spic", "chink", "kike",
    "tranny", "trans", "transgender", "muslims", "islamist", "islamophobia", "jews", "jewish", "anti-semitic",
    "blacks", "whites", "immigrants", "mexicans", "homosexual", "homophobia", "nazi", "nazis", "gypsy"
}

INSULT_WORDS = {
    "idiot", "idiots", "idiotic", "moron", "morons", "stupid", "stupidity", "fool", "fools",
    "foolish", "clown", "clowns", "loser", "losers", "scumbag", "scumbags", "dumb", "dumber",
    "dumbest", "jerk", "jerks", "buffoon", "buffoons", "ignorant", "ignorance", "disgusting",
    "pathetic", "ugly", "hypocrite", "hypocrites", "liar", "liars", "cheat", "corrupt", "trash",
    "garbage", "pig", "pigs", "filth", "filthy", "incompetent", "imbecile", "useless", "coward",
    "cowards", "shame", "scum", "creep", "freak"
}

def get_token_aspect(token_lower: str, doc_category: str) -> str:
    if token_lower in THREAT_WORDS:
        return "THREAT"
    if token_lower in OBSCENE_WORDS:
        return "OBSCENE"
    if token_lower in IDENTITY_WORDS:
        return "IDENTITY_ATTACK"
    if token_lower in INSULT_WORDS:
        return "INSULT"
    return doc_category


MULTI_WORD_PATTERNS = {
    "THREAT": [
        "kill you", "kill yourself", "kill them", "die in a fire", "die in a", "burn in hell", "burn you",
        "rot in hell", "hang yourself", "blow your head off", "blow your", "shoot you", "shoot them",
        "beat you", "beat the shit out of", "beat the shit", "slit your throat", "slit your", "stab you",
        "drop dead", "break your neck", "end your life", "choke you", "strangle you"
    ],
    "OBSCENE": [
        "fuck you", "fuck off", "shut the fuck up", "shut up", "piece of shit", "eat shit", "full of shit",
        "son of a bitch", "dumb as fuck", "get the fuck out", "get the fuck", "mother fucker", "holy shit",
        "dip shit", "little bitch", "bitch ass", "kiss my ass"
    ],
    "IDENTITY_ATTACK": [
        "left wing nut", "left wing", "right wing nut", "right wing", "white trash", "dirty jew",
        "dirty arab", "fucking gay", "retarded idiot", "islamic terrorist", "illegal immigrant",
        "go back to your country"
    ],
    "OTHER_TOXIC": [
        "waste of space", "waste of oxygen", "waste of time", "waste of breath", "scum of the earth",
        "troll account", "toxic behavior", "complete garbage", "brain dead", "mentally ill",
        "sick in the head", "get a life", "seek help", "delusional fool"
    ],
    "INSULT": [
        "arrogant jerk", "ignorant fool", "stupid clown", "big sissy", "disgusting rat",
        "lying cheat", "corrupt pig", "pathetic loser", "total fraud", "useless coward"
    ]
}

def align_doc_to_bio(doc, positions: set, category: str) -> List[str]:
    bio_tags = ["O"] * len(doc)
    doc_text_lower = doc.text.lower()
    
    if category != "NEUTRAL_O" and positions:
        # Step 1: Detect multi-word phrase patterns to generate natural B- and I- sequence continuations
        matched_tokens = set()
        for cat, phrases in MULTI_WORD_PATTERNS.items():
            for phrase in phrases:
                start_char = 0
                while True:
                    found_idx = doc_text_lower.find(phrase, start_char)
                    if found_idx == -1:
                        break
                    phrase_end = found_idx + len(phrase)
                    phrase_char_indices = set(range(found_idx, phrase_end))
                    
                    # If this multi-word phrase intersects with toxic annotations or category matches
                    if phrase_char_indices.intersection(positions) or category == cat:
                        # Find tokens overlapping with this phrase
                        phrase_tokens = []
                        for t_idx, token in enumerate(doc):
                            t_len = len(token.text)
                            if t_len > 0 and not token.text.isspace():
                                t_chars = set(range(token.idx, token.idx + t_len))
                                if t_chars.intersection(phrase_char_indices):
                                    phrase_tokens.append(t_idx)
                        
                        if phrase_tokens:
                            first = phrase_tokens[0]
                            bio_tags[first] = f"B-{cat}"
                            matched_tokens.add(first)
                            for rest in phrase_tokens[1:]:
                                bio_tags[rest] = f"I-{cat}"
                                matched_tokens.add(rest)
                                
                    start_char = found_idx + len(phrase)

        # Step 2: For remaining tokens, align single-word toxic positions
        in_span = False
        curr_cat = category
        for t_idx, token in enumerate(doc):
            if t_idx in matched_tokens:
                in_span = False
                continue
                
            token_len = len(token.text)
            if token_len == 0 or token.text.isspace():
                bio_tags[t_idx] = "O"
                in_span = False
                continue
                
            token_chars = set(range(token.idx, token.idx + token_len))
            overlap = token_chars.intersection(positions)
            is_token_toxic = (len(overlap) >= token_len * 0.5) if token_len > 1 else (len(overlap) > 0)
            
            if is_token_toxic:
                tok_cat = get_token_aspect(token.text.lower(), category)
                if not in_span:
                    curr_cat = tok_cat
                    bio_tags[t_idx] = f"B-{curr_cat}"
                    in_span = True
                else:
                    if tok_cat != curr_cat and tok_cat in {"THREAT", "OBSCENE", "IDENTITY_ATTACK", "INSULT"}:
                        curr_cat = tok_cat
                        bio_tags[t_idx] = f"B-{curr_cat}"
                    else:
                        bio_tags[t_idx] = f"I-{curr_cat}"
            else:
                bio_tags[t_idx] = "O"
                in_span = False
                
    return bio_tags

def load_and_group_raw(file_path: Path) -> Dict[str, List[Dict[str, Any]]]:
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

def create_non_duplicated_splits(
    grouped_train: Dict[str, List[Dict[str, Any]]],
    grouped_test: Dict[str, List[Dict[str, Any]]],
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Creates Train, Val, Test splits with:
    - ZERO duplicate samples (no oversampling).
    - Downsampled majority classes (INSULT downsampled by ~68%, NEUTRAL downsampled by ~93%)
      to prevent 'O' from overwhelming the sequence model.
    - All available unique minority samples (THREAT, IDENTITY_ATTACK, OTHER_TOXIC) used fully.
    """
    random.seed(seed)
    
    # Train targets (exact unique samples taken without replacement)
    # Val reservation counts (taken from train pool to ensure clean out-of-fold validation)
    val_counts = {
        "THREAT": 40,
        "IDENTITY_ATTACK": 80,
        "OBSCENE": 80,
        "OTHER_TOXIC": 50,
        "INSULT": 120,
        "NEUTRAL_O": 60
    }
    
    # Majority class caps for training to prevent O and INSULT explosion
    train_caps = {
        "INSULT": 950,      # Downsample 2839 -> 950
        "NEUTRAL_O": 350    # Downsample 4878 -> 350 to prevent O domination!
    }
    
    train_samples_raw = []
    val_samples_raw = []
    
    for cat in PRIORITY_CATEGORIES + ["NEUTRAL_O"]:
        pool = list(grouped_train.get(cat, []))
        random.shuffle(pool)
        
        v_num = min(val_counts.get(cat, 50), len(pool) // 5)
        val_items = pool[:v_num]
        train_pool = pool[v_num:]
        
        # Apply cap if defined, otherwise take ALL remaining unique samples
        if cat in train_caps and len(train_pool) > train_caps[cat]:
            train_items = train_pool[:train_caps[cat]]
        else:
            train_items = train_pool
            
        for item in train_items:
            cp = dict(item)
            cp["assigned_category"] = cat
            train_samples_raw.append(cp)
            
        for item in val_items:
            cp = dict(item)
            cp["assigned_category"] = cat
            val_samples_raw.append(cp)
            
    # Test split from grouped_test (completely independent 1,000 raw samples, ZERO duplicates)
    test_samples_raw = []
    test_caps = {
        "INSULT": 120,
        "NEUTRAL_O": 80
    }
    for cat in PRIORITY_CATEGORIES + ["NEUTRAL_O"]:
        pool = list(grouped_test.get(cat, []))
        random.shuffle(pool)
        if cat in test_caps and len(pool) > test_caps[cat]:
            selected = pool[:test_caps[cat]]
        else:
            selected = pool
            
        for item in selected:
            cp = dict(item)
            cp["assigned_category"] = cat
            test_samples_raw.append(cp)
            
    random.shuffle(train_samples_raw)
    random.shuffle(val_samples_raw)
    random.shuffle(test_samples_raw)
    
    return train_samples_raw, val_samples_raw, test_samples_raw

def process_and_align_samples(raw_samples: List[Dict[str, Any]], nlp) -> List[Dict[str, Any]]:
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
        
    train_raw, val_raw, test_raw = create_non_duplicated_splits(grouped_train, grouped_test, seed=42)
    
    logging.info(f"Processing {len(train_raw)} unique train samples (zero duplicates, downsampled majority)...")
    train_samples = process_and_align_samples(train_raw, nlp)
    
    logging.info(f"Processing {len(val_raw)} unique val samples...")
    val_samples = process_and_align_samples(val_raw, nlp)
    
    logging.info(f"Processing {len(test_raw)} unique test samples...")
    test_samples = process_and_align_samples(test_raw, nlp)
    
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
        "dataset_name": "Multi-Aspect Toxic Spans (Non-duplicated, Balanced Ratio)",
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
        
    print("\n--- CLEAN TRAIN TAG DISTRIBUTION ---")
    for tag, cnt in train_tag_counts.most_common():
        print(f"  {tag:<22}: {cnt}")
        
    print("\n--- CLEAN TEST TAG DISTRIBUTION ---")
    for tag, cnt in test_tag_counts.most_common():
        print(f"  {tag:<22}: {cnt}")

if __name__ == "__main__":
    main()
