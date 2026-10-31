"""
Hidden Markov Model (HMM) for Sequence Labeling / Token Classification.
Implements MLE transition/emission probabilities with strict BIO transition constraints
and unigram-prior smoothed emission probabilities for robust Viterbi decoding.
"""

from collections import defaultdict, Counter
import numpy as np
from typing import List, Dict, Tuple

class HiddenMarkovModel:
    def __init__(self, alpha: float = 1e-4, beta: float = 0.01):
        """
        alpha: Smoothing for valid transitions
        beta: Weight for unigram emission smoothing
        """
        self.alpha = alpha
        self.beta = beta
        
        self.tag_counts = Counter()
        self.word_counts = Counter()
        self.transition_counts = defaultdict(Counter)
        self.emission_counts = defaultdict(Counter)
        self.start_counts = Counter()
        
        self.states = []
        self.state_to_ix = {}
        self.vocab = set()
        self.p_word_prior = {}
        
    def fit(self, sentences_tokens: List[List[str]], sentences_tags: List[List[str]]):
        """Estimate transition and emission probabilities from training sequences."""
        total_tokens = 0
        for tokens, tags in zip(sentences_tokens, sentences_tags):
            if not tokens or not tags:
                continue
            
            self.start_counts[tags[0]] += 1
            
            for t in range(len(tokens)):
                word = tokens[t].lower()
                tag = tags[t]
                
                self.tag_counts[tag] += 1
                self.word_counts[word] += 1
                self.vocab.add(word)
                self.emission_counts[tag][word] += 1
                total_tokens += 1
                
                if t > 0:
                    prev_tag = tags[t - 1]
                    self.transition_counts[prev_tag][tag] += 1
                    
        self.states = sorted(list(self.tag_counts.keys()))
        self.state_to_ix = {s: i for i, s in enumerate(self.states)}
        self.p_word_prior = {w: c / total_tokens for w, c in self.word_counts.items()}
        
    def get_transition_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Return transition probability matrix A as a 2D numpy array.
        Enforces strict BIO sequence constraints:
        - O -> I-X is impossible
        - B-X -> I-Y (X != Y) is impossible
        - I-X -> I-Y (X != Y) is impossible
        """
        num_states = len(self.states)
        trans_matrix = np.zeros((num_states, num_states))
        
        for i, s_prev in enumerate(self.states):
            denom = self.tag_counts[s_prev] + self.alpha * num_states
            for j, s_curr in enumerate(self.states):
                # Check BIO validity
                is_invalid = False
                if s_curr.startswith("I-"):
                    curr_cat = s_curr[2:]
                    if s_prev == "O":
                        is_invalid = True
                    elif s_prev.startswith("B-") or s_prev.startswith("I-"):
                        if s_prev[2:] != curr_cat:
                            is_invalid = True
                            
                if is_invalid:
                    trans_matrix[i, j] = 1e-12
                else:
                    num = self.transition_counts[s_prev][s_curr] + self.alpha
                    trans_matrix[i, j] = num / denom
                    
            # Row normalization
            row_sum = trans_matrix[i].sum()
            if row_sum > 0:
                trans_matrix[i] /= row_sum
                
        return trans_matrix, self.states

    def predict_sentence(self, tokens: List[str]) -> List[str]:
        """Viterbi decoding for a single sentence with BIO-consistent transitions."""
        if not tokens:
            return []
            
        N = len(self.states)
        T = len(tokens)
        
        viterbi = np.zeros((T, N))
        backpointer = np.zeros((T, N), dtype=int)
        
        A, _ = self.get_transition_matrix()
        log_A = np.log(A)
        
        # Initialization (t = 0)
        total_starts = sum(self.start_counts.values()) + 1e-4 * N
        w0 = tokens[0].lower()
        pw0 = self.p_word_prior.get(w0, 1e-6)
        
        for j, s in enumerate(self.states):
            # START cannot transition into I- tags
            if s.startswith("I-"):
                viterbi[0, j] = -1e9
                continue
                
            p_start = (self.start_counts[s] + 1e-4) / total_starts
            p_emit = (self.emission_counts[s][w0] + self.beta * pw0) / (self.tag_counts[s] + self.beta)
            viterbi[0, j] = np.log(p_start + 1e-12) + np.log(p_emit + 1e-12)
            
        # Recursion (t = 1 .. T-1)
        for t in range(1, T):
            wt = tokens[t].lower()
            pwt = self.p_word_prior.get(wt, 1e-6)
            for j, s_curr in enumerate(self.states):
                p_emit = (self.emission_counts[s_curr][wt] + self.beta * pwt) / (self.tag_counts[s_curr] + self.beta)
                log_emit = np.log(p_emit + 1e-12)
                
                candidates = viterbi[t - 1] + log_A[:, j]
                best_prev = np.argmax(candidates)
                
                viterbi[t, j] = candidates[best_prev] + log_emit
                backpointer[t, j] = best_prev
                
        # Termination
        best_last_state = np.argmax(viterbi[T - 1])
        
        # Path backtracking
        best_path = [best_last_state]
        for t in range(T - 1, 0, -1):
            prev_state = backpointer[t, best_path[-1]]
            best_path.append(prev_state)
            
        best_path.reverse()
        return [self.states[i] for i in best_path]

    def predict(self, sentences: List[List[str]]) -> List[List[str]]:
        """Predict tags for a list of tokenized sentences."""
        return [self.predict_sentence(tokens) for tokens in sentences]
