"""
Feature-based Linear-Chain CRF for Sequence Labeling.
Extracts orthographic, linguistic, context-window, and POS features.
Uses sklearn-crfsuite with L-BFGS optimization.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import sklearn_crfsuite

def word2features(sent_tokens: List[str], sent_pos: List[str], i: int) -> Dict[str, Any]:
    """Extract clean orthographic and contextual features for token at index i."""
    word = sent_tokens[i]
    pos = sent_pos[i] if sent_pos else "UNK"

    features = {
        "bias": 1.0,
        "word_lower": word.lower(),
        "suffix_3": word[-3:] if len(word) >= 3 else word,
        "suffix_2": word[-2:] if len(word) >= 2 else word,
        "prefix_3": word[:3] if len(word) >= 3 else word,
        "prefix_2": word[:2] if len(word) >= 2 else word,
        "is_upper": word.isupper(),
        "is_title": word.istitle(),
        "is_digit": word.isdigit(),
        "has_punct": any(c in "!@#$%^&*()-_=+[]{}|;:'\",.<>?/" for c in word),
        "postag": pos,
    }
    
    # Previous token features
    if i > 0:
        prev_word = sent_tokens[i - 1]
        prev_pos = sent_pos[i - 1] if sent_pos else "UNK"
        features.update({
            "prev_word_lower": prev_word.lower(),
            "prev_is_title": prev_word.istitle(),
            "prev_is_upper": prev_word.isupper(),
            "prev_postag": prev_pos,
        })
    else:
        features["BOS"] = True

    # Next token features
    if i < len(sent_tokens) - 1:
        next_word = sent_tokens[i + 1]
        next_pos = sent_pos[i + 1] if sent_pos else "UNK"
        features.update({
            "next_word_lower": next_word.lower(),
            "next_is_title": next_word.istitle(),
            "next_is_upper": next_word.isupper(),
            "next_postag": next_pos,
        })
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

    def fit(self, sentences_tokens: List[List[str]], sentences_pos: List[List[str]], sentences_tags: List[List[str]]):
        X = [sent2features(tokens, pos) for tokens, pos in zip(sentences_tokens, sentences_pos)]
        y = sentences_tags
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
        except Exception as e:
            print(f"Note: transition_features_ extraction bypassed ({e}).")
                
        return matrix, labels
