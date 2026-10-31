"""
Hidden Markov Model (HMM) for Sequence Labeling / Token Classification.
Implements MLE transition/emission probabilities with Laplace smoothing
and dynamic programming Viterbi decoding.
"""

from collections import defaultdict, Counter
import numpy as np
from typing import List, Dict, Tuple

class HiddenMarkovModel:
    def __init__(self, alpha: float = 1e-3, beta: float = 1e-3):
        """
        alpha: Smoothing parameter for transition probabilities
        beta: Smoothing parameter for emission probabilities
        """
        self.alpha = alpha
        self.beta = beta
        
        self.tag_counts = Counter()
        self.word_counts = Counter()
        self.transition_counts = defaultdict(Counter) # C(y_{t-1}, y_t)
        self.emission_counts = defaultdict(Counter)   # C(y_t, x_t)
        self.start_counts = Counter()                 # C(START, y_1)
        
        self.states = []
        self.state_to_ix = {}
        self.vocab = set()
        
    def fit(self, sentences_tokens: List[List[str]], sentences_tags: List[List[str]]):
        """Estimate transition and emission probabilities from training sequences."""
        for tokens, tags in zip(sentences_tokens, sentences_tags):
            if not tokens or not tags:
                continue
            
            # Start tag
            self.start_counts[tags[0]] += 1
            
            for t in range(len(tokens)):
                word = tokens[t].lower()
                tag = tags[t]
                
                self.tag_counts[tag] += 1
                self.word_counts[word] += 1
                self.vocab.add(word)
                self.emission_counts[tag][word] += 1
                
                if t > 0:
                    prev_tag = tags[t - 1]
                    self.transition_counts[prev_tag][tag] += 1
                    
        self.states = sorted(list(self.tag_counts.keys()))
        self.state_to_ix = {s: i for i, s in enumerate(self.states)}
        
    def get_transition_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Return transition probability matrix A as a 2D numpy array."""
        num_states = len(self.states)
        trans_matrix = np.zeros((num_states, num_states))
        
        for i, s_prev in enumerate(self.states):
            denom = self.tag_counts[s_prev] + self.alpha * num_states
            for j, s_curr in enumerate(self.states):
                num = self.transition_counts[s_prev][s_curr] + self.alpha
                trans_matrix[i, j] = num / denom
                
        return trans_matrix, self.states

    def predict_sentence(self, tokens: List[str]) -> List[str]:
        """Viterbi decoding for a single sentence."""
        if not tokens:
            return []
            
        N = len(self.states)
        T = len(tokens)
        
        # Log probabilities to prevent underflow
        viterbi = np.zeros((T, N))
        backpointer = np.zeros((T, N), dtype=int)
        
        # Initialization (t = 0)
        total_starts = sum(self.start_counts.values()) + self.alpha * N
        w0 = tokens[0].lower()
        for j, s in enumerate(self.states):
            p_start = (self.start_counts[s] + self.alpha) / total_starts
            p_emit = (self.emission_counts[s][w0] + self.beta) / (self.tag_counts[s] + self.beta * (len(self.vocab) + 1))
            viterbi[0, j] = np.log(p_start + 1e-12) + np.log(p_emit + 1e-12)
            
        # Transition matrix in log space
        A, _ = self.get_transition_matrix()
        log_A = np.log(A + 1e-12)
        
        # Recursion (t = 1 .. T-1)
        for t in range(1, T):
            wt = tokens[t].lower()
            for j, s_curr in enumerate(self.states):
                p_emit = (self.emission_counts[s_curr][wt] + self.beta) / (self.tag_counts[s_curr] + self.beta * (len(self.vocab) + 1))
                log_emit = np.log(p_emit + 1e-12)
                
                # Max over previous states
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

