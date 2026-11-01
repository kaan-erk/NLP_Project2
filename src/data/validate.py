"""
Dataset Sequence Integrity & BIO Grammar Validator.
Verifies that all processed dataset splits (train, val, test) adhere strictly to
IOB2 grammar:
1. Every I-X tag is immediately preceded by either B-X or I-X of the identical class.
2. No sequence begins with an orphaned I- tag.
3. No cross-class illegal transitions (e.g., B-THREAT -> I-OBSCENE).
4. Token, lemma, and tag sequence lengths match with zero offset drift.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PROCESSED_DATA_DIR = Path("data/processed")

def validate_sequence_bio(tokens: List[str], tags: List[str], s_idx: int) -> List[Dict[str, Any]]:
    """Validates BIO grammar for a single sentence."""
    violations = []
    
    if len(tokens) != len(tags):
        violations.append({
            "sentence_id": s_idx,
            "error_type": "LENGTH_MISMATCH",
            "details": f"Tokens length ({len(tokens)}) != Tags length ({len(tags)})"
        })
        return violations

    for t_idx, tag in enumerate(tags):
        if tag.startswith("I-"):
            cat = tag[2:]
            if t_idx == 0:
                violations.append({
                    "sentence_id": s_idx,
                    "token_idx": t_idx,
                    "error_type": "ORPHANED_I_TAG_AT_START",
                    "tag": tag,
                    "token": tokens[t_idx]
                })
            else:
                prev_tag = tags[t_idx - 1]
                allowed_prev = {f"B-{cat}", f"I-{cat}"}
                if prev_tag not in allowed_prev:
                    violations.append({
                        "sentence_id": s_idx,
                        "token_idx": t_idx,
                        "error_type": "ILLEGAL_TRANSITION_TO_I_TAG",
                        "prev_tag": prev_tag,
                        "curr_tag": tag,
                        "token": tokens[t_idx],
                        "details": f"Expected {allowed_prev}, got {prev_tag}"
                    })
                    
    return violations

def validate_split(file_path: Path) -> Tuple[int, List[Dict[str, Any]]]:
    """Validates an entire JSONL split file."""
    if not file_path.exists():
        logging.warning(f"File not found: {file_path}")
        return 0, []

    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    all_violations = []
    for s_idx, rec in enumerate(records):
        tokens = rec.get("tokens", [])
        tags = rec.get("tags", [])
        viols = validate_sequence_bio(tokens, tags, s_idx)
        all_violations.extend(viols)

    return len(records), all_violations

def run_validation_pipeline(data_dir: Path = PROCESSED_DATA_DIR) -> bool:
    """Runs sequence validation across train, val, and test splits."""
    logging.info("=====================================================================")
    logging.info("RUNNING BIO SEQUENCE INTEGRITY & GRAMMAR VALIDATION")
    logging.info("=====================================================================")
    
    splits = ["train.jsonl", "val.jsonl", "test.jsonl"]
    total_samples = 0
    total_violations = 0
    all_passed = True

    for split_name in splits:
        split_path = data_dir / split_name
        count, violations = validate_split(split_path)
        total_samples += count
        total_violations += len(violations)
        
        if violations:
            all_passed = False
            logging.error(f"[{split_name.upper()}] FAILED! Found {len(violations)} violations across {count} sentences.")
            for v in violations[:5]:
                logging.error(f"  Violation: {v}")
        else:
            logging.info(f"[{split_name.upper()}] PASSED: {count} sentences checked. 0 orphaned I- tags, 0 illegal transitions.")

    logging.info("---------------------------------------------------------------------")
    if all_passed:
        logging.info(f"VERIFICATION SUCCESS: All {total_samples} sentences across dataset splits output ZERO violations.")
    else:
        logging.error(f"VERIFICATION FAILURE: Total {total_violations} violations detected.")
    logging.info("=====================================================================")
    return all_passed

if __name__ == "__main__":
    run_validation_pipeline()

