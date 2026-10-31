"""
PyTorch Dataset and Collate Function for Token Classification.
Handles sequence padding, vocabulary lookup, and attention masks.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
from torch.utils.data import Dataset

class ToxicSpanDataset(Dataset):
    def __init__(self, jsonl_file: Path, word_to_ix: Dict[str, int], tag_to_ix: Dict[str, int]):
        self.samples = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
                    
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
        self.unk_idx = word_to_ix.get("<UNK>", 1)
        self.pad_word_idx = word_to_ix.get("<PAD>", 0)
        self.pad_tag_idx = tag_to_ix.get("<PAD>", 0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        tokens = item["tokens"]
        tags = item["tags"]
        pos_tags = item.get("pos_tags", [])

        word_ids = [self.word_to_ix.get(t.lower(), self.unk_idx) for t in tokens]
        tag_ids = [self.tag_to_ix.get(tag, self.tag_to_ix.get("O", 0)) for tag in tags]

        return {
            "tokens": tokens,
            "word_ids": torch.tensor(word_ids, dtype=torch.long),
            "tag_ids": torch.tensor(tag_ids, dtype=torch.long),
            "tags": tags,
            "pos_tags": pos_tags
        }

def pad_collate_fn(batch: List[Dict[str, Any]], pad_word_idx: int = 0, pad_tag_idx: int = 0) -> Dict[str, Any]:
    """
    Pads sequences within the batch to uniform length and creates binary mask.
    """
    word_seqs = [item["word_ids"] for item in batch]
    tag_seqs = [item["tag_ids"] for item in batch]
    raw_tokens = [item["tokens"] for item in batch]
    raw_tags = [item["tags"] for item in batch]
    raw_pos = [item["pos_tags"] for item in batch]

    lengths = torch.tensor([len(s) for s in word_seqs], dtype=torch.long)
    max_len = max(len(s) for s in word_seqs)

    padded_words = torch.full((len(batch), max_len), pad_word_idx, dtype=torch.long)
    padded_tags = torch.full((len(batch), max_len), pad_tag_idx, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.float32)

    for i, (words, tags) in enumerate(zip(word_seqs, tag_seqs)):
        seq_len = len(words)
        padded_words[i, :seq_len] = words
        padded_tags[i, :seq_len] = tags
        mask[i, :seq_len] = 1.0

    return {
        "words": padded_words,
        "tags": padded_tags,
        "mask": mask,
        "lengths": lengths,
        "raw_tokens": raw_tokens,
        "raw_tags": raw_tags,
        "raw_pos": raw_pos
    }

