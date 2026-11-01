"""
Logistic Regression Token Classification Baseline.
Classifies tokens independently using local lexical and syntactic features,
with smoothed inverse square root class weighting.
"""

from typing import List, Dict, Any
from collections import Counter
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def word2features(sent_tokens: List[str], sent_pos: List[str], i: int) -> Dict[str, Any]:
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
        self.max_iter = max_iter
        self.C = C
        self.clf = None
        self.classes_ = []

    def fit(self, sentences_tokens: List[List[str]], sentences_pos: List[List[str]], sentences_tags: List[List[str]]):
        all_features = []
        all_tags = []

        for tokens, pos_seq, tags in zip(sentences_tokens, sentences_pos, sentences_tags):
            for i in range(len(tokens)):
                all_features.append(word2features(tokens, pos_seq, i))
                all_tags.append(tags[i])

        X = self.vectorizer.fit_transform(all_features)
        y = np.array(all_tags)
        self.classes_ = sorted(list(set(all_tags)))

        # Smoothed Inverse Square Root Class Weights (w_c = 1 / sqrt(freq_c))
        tag_counts = Counter(all_tags)
        raw_weights = {cls: 1.0 / np.sqrt(tag_counts.get(cls, 1)) for cls in self.classes_}
        mean_w = np.mean(list(raw_weights.values()))
        smoothed_weights = {cls: float(np.clip(w / mean_w, 0.2, 5.0)) for cls, w in raw_weights.items()}

        self.clf = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=smoothed_weights,
            solver="lbfgs",
            random_state=42
        )
        self.clf.fit(X, y)

    def predict(self, sentences_tokens: List[List[str]], sentences_pos: List[List[str]] = None) -> List[List[str]]:
        predictions = []
        for i, tokens in enumerate(sentences_tokens):
            if not tokens:
                predictions.append([])
                continue
            pos_seq = sentences_pos[i] if sentences_pos else []
            feat_list = [word2features(tokens, pos_seq, t) for t in range(len(tokens))]
            X_sent = self.vectorizer.transform(feat_list)
            preds = self.clf.predict(X_sent)
            predictions.append(list(preds))
        return predictions
