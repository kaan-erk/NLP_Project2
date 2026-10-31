"""
Token-level Logistic Regression Baseline for Sequence Labeling.
Classifies each token independently without sequence transition modeling.
Serves as an essential non-sequence baseline to demonstrate the value of HMM/CRF/BiLSTM.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def extract_token_features(sent_tokens: List[str], sent_pos: List[str], i: int) -> Dict[str, Any]:
    """Extract orthographic and local window features for a single token."""
    word = sent_tokens[i]
    pos = sent_pos[i] if sent_pos else "UNK"

    feat = {
        "bias": 1.0,
        "word_lower": word.lower(),
        "suffix_3": word[-3:] if len(word) >= 3 else word,
        "suffix_2": word[-2:] if len(word) >= 2 else word,
        "prefix_3": word[:3] if len(word) >= 3 else word,
        "is_upper": float(word.isupper()),
        "is_title": float(word.istitle()),
        "is_digit": float(word.isdigit()),
        "postag": pos,
    }

    if i > 0:
        feat["prev_word_lower"] = sent_tokens[i - 1].lower()
        feat["prev_postag"] = sent_pos[i - 1] if sent_pos else "UNK"
    else:
        feat["BOS"] = 1.0

    if i < len(sent_tokens) - 1:
        feat["next_word_lower"] = sent_tokens[i + 1].lower()
        feat["next_postag"] = sent_pos[i + 1] if sent_pos else "UNK"
    else:
        feat["EOS"] = 1.0

    return feat

class LogisticRegressionBaseline:
    def __init__(self, max_iter: int = 50, C: float = 1.0):
        self.vectorizer = DictVectorizer(sparse=True)
        self.clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42
        )
        self.classes_ = []

    def fit(self, sentences_tokens: List[List[str]], sentences_pos: List[List[str]], sentences_tags: List[List[str]]):
        flat_features = []
        flat_labels = []

        for tokens, pos, tags in zip(sentences_tokens, sentences_pos, sentences_tags):
            for i in range(len(tokens)):
                flat_features.append(extract_token_features(tokens, pos, i))
                flat_labels.append(tags[i])

        X = self.vectorizer.fit_transform(flat_features)
        self.clf.fit(X, flat_labels)
        self.classes_ = list(self.clf.classes_)

    def predict(self, sentences_tokens: List[List[str]], sentences_pos: List[List[str]]) -> List[List[str]]:
        flat_features = []
        sentence_lengths = []

        for tokens, pos in zip(sentences_tokens, sentences_pos):
            sentence_lengths.append(len(tokens))
            for i in range(len(tokens)):
                flat_features.append(extract_token_features(tokens, pos, i))

        if not flat_features:
            return [[] for _ in sentences_tokens]

        X = self.vectorizer.transform(flat_features)
        flat_preds = self.clf.predict(X)

        predicted_sequences = []
        offset = 0
        for length in sentence_lengths:
            predicted_sequences.append(list(flat_preds[offset : offset + length]))
            offset += length

        return predicted_sequences
