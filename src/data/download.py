"""
Script to download raw dataset from Hugging Face and store it in data/raw/.
Dataset: heegyu/toxic-spans (SemEval-2021 Task 5 with fine-grained toxicity types)
"""

import json
import os
from pathlib import Path
from datasets import load_dataset

RAW_DATA_DIR = Path("data/raw")

def download_data():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading 'heegyu/toxic-spans' from Hugging Face...")
    ds = load_dataset("heegyu/toxic-spans")

    train_path = RAW_DATA_DIR / "toxic_spans_train_raw.jsonl"
    test_path = RAW_DATA_DIR / "toxic_spans_test_raw.jsonl"

    print(f"Saving raw train data to {train_path} ({len(ds['train'])} examples)...")
    with open(train_path, "w", encoding="utf-8") as f:
        for item in ds["train"]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saving raw test data to {test_path} ({len(ds['test'])} examples)...")
    with open(test_path, "w", encoding="utf-8") as f:
        for item in ds["test"]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Data download completed successfully!")

if __name__ == "__main__":
    download_data()

