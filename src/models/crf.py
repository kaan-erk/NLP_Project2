"""
Feature-based Linear-Chain CRF for Sequence Labeling.
Extracts orthographic, linguistic, context-window, and POS features.
Uses sklearn-crfsuite with L-BFGS optimization and robust feature string escaping.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from collections import defaultdict
import sklearn_crfsuite

def clean_feat_val(val: Any) -> str:
    """Escapes newlines, control characters, and spaces for crfsuite dump parser safety."""
    if isinstance(val, str):
        return val.replace("\n", "\\n").replace("\r", "\\r").replace(" ", "_")
    return val

def word2features(sent_tokens: List[str], sent_pos: List[str], i: int) -> Dict[str, Any]:
    """Extract clean orthographic and contextual features for token at index i."""
    raw_word = sent_tokens[i]
    word = clean_feat_val(raw_word)
    pos = clean_feat_val(sent_pos[i] if sent_pos else "UNK")

    features = {
        "bias": 1.0,
        "word_lower": word.lower(),
        "suffix_3": word[-3:] if len(word) >= 3 else word,
        "suffix_2": word[-2:] if len(word) >= 2 else word,
        "prefix_3": word[:3] if len(word) >= 3 else word,
        "prefix_2": word[:2] if len(word) >= 2 else word,
        "is_upper": raw_word.isupper(),
        "is_title": raw_word.istitle(),
        "is_digit": raw_word.isdigit(),
        "has_punct": any(c in "!@#$%^&*()-_=+[]{}|;:'\",.<>?/" for c in raw_word),
        "postag": pos,
    }
    
    # Previous token features
    if i > 0:
        prev_raw = sent_tokens[i - 1]
        prev_word = clean_feat_val(prev_raw)
        prev_pos = clean_feat_val(sent_pos[i - 1] if sent_pos else "UNK")
        features.update({
            "prev_word_lower": prev_word.lower(),
            "prev_is_title": prev_raw.istitle(),
            "prev_is_upper": prev_raw.isupper(),
            "prev_postag": prev_pos,
            "prev_plus_curr": f"{prev_word.lower()}_{word.lower()}",
        })
        if i > 1:
            prev2_word = clean_feat_val(sent_tokens[i - 2])
            features["prev2_word_lower"] = prev2_word.lower()
    else:
        features["BOS"] = True

    # Next token features
    if i < len(sent_tokens) - 1:
        next_raw = sent_tokens[i + 1]
        next_word = clean_feat_val(next_raw)
        next_pos = clean_feat_val(sent_pos[i + 1] if sent_pos else "UNK")
        features.update({
            "next_word_lower": next_word.lower(),
            "next_is_title": next_raw.istitle(),
            "next_is_upper": next_raw.isupper(),
            "next_postag": next_pos,
            "curr_plus_next": f"{word.lower()}_{next_word.lower()}",
        })
        if i < len(sent_tokens) - 2:
            next2_word = clean_feat_val(sent_tokens[i + 2])
            features["next2_word_lower"] = next2_word.lower()
    else:
        features["EOS"] = True

    return features

def sent2features(sent_tokens: List[str], sent_pos: List[str]) -> List[Dict[str, Any]]:
    return [word2features(sent_tokens, sent_pos, i) for i in range(len(sent_tokens))]

class CRFModel:
    def __init__(self, c1: float = 0.1, c2: float = 0.1, max_iterations: int = 50):
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True,
            verbose=False
        )
        self.classes_ = []
        self._empirical_transitions = defaultdict(lambda: defaultdict(int))

    def fit(self, sentences_tokens: List[List[str]], sentences_pos: List[List[str]], sentences_tags: List[List[str]]):
        X = [sent2features(tokens, pos) for tokens, pos in zip(sentences_tokens, sentences_pos)]
        y = sentences_tags
        
        # Track transitions from data for robust fallback
        for tag_seq in y:
            for t in range(1, len(tag_seq)):
                self._empirical_transitions[tag_seq[t - 1]][tag_seq[t]] += 1
                
        self.model.fit(X, y)
        self.classes_ = list(self.model.classes_)

    def predict(self, sentences_tokens: List[List[str]], sentences_pos: List[List[str]]) -> List[List[str]]:
        X = [sent2features(tokens, pos) for tokens, pos in zip(sentences_tokens, sentences_pos)]
        return self.model.predict(X)

    def get_transition_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Extract learned transition weights from the trained CRF model."""
        labels = sorted(self.classes_)
        n_labels = len(labels)
        matrix = np.zeros((n_labels, n_labels))
        
        try:
            transitions = self.model.transition_features_
            for i, from_tag in enumerate(labels):
                for j, to_tag in enumerate(labels):
                    matrix[i, j] = transitions.get((from_tag, to_tag), 0.0)
        except Exception:
            # Safe Fallback: compute empirical transition probability log-odds
            for i, from_tag in enumerate(labels):
                total = sum(self._empirical_transitions[from_tag].values()) + 1e-6
                for j, to_tag in enumerate(labels):
                    count = self._empirical_transitions[from_tag][to_tag]
                    matrix[i, j] = np.log((count + 1e-4) / total)
                
        return matrix, labels
