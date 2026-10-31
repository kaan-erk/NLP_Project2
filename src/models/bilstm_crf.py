"""
PyTorch BiLSTM-CRF Sequence Labeling Model.
Combines bidirectional LSTM contextual representations with a Linear-Chain
Conditional Random Field (CRF) layer and exact Viterbi decoding.
"""

from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn

def log_sum_exp(vec: torch.Tensor) -> torch.Tensor:
    """Numerically stable log-sum-exp over the last dimension."""
    max_score = vec.max(-1, keepdim=True).values
    return max_score.squeeze(-1) + torch.log(torch.sum(torch.exp(vec - max_score), dim=-1))

class BiLSTM_CRF(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        tag_to_ix: Dict[str, int],
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        dropout_p: float = 0.3,
        pad_idx: int = 0
    ):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.ix_to_tag = {v: k for k, v in tag_to_ix.items()}
        self.num_tags = len(tag_to_ix)
        self.pad_idx = pad_idx

        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        self.PAD_TAG = "<PAD>"
        
        self.start_idx = tag_to_ix[self.START_TAG]
        self.stop_idx = tag_to_ix[self.STOP_TAG]
        self.pad_tag_idx = tag_to_ix.get(self.PAD_TAG, -1)

        # Layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.hidden2tag = nn.Linear(hidden_dim, self.num_tags)

        # Transition parameters: transitions[i, j] is score of transitioning from j to i
        self.transitions = nn.Parameter(torch.randn(self.num_tags, self.num_tags))

        # Enforce structural constraints
        # Impossible transitions get large negative score
        self.transitions.data[self.start_idx, :] = -10000.0  # Cannot transition TO start
        self.transitions.data[:, self.stop_idx] = -10000.0   # Cannot transition FROM stop
        if self.pad_tag_idx != -1:
            self.transitions.data[self.pad_tag_idx, :] = -10000.0
            self.transitions.data[:, self.pad_tag_idx] = -10000.0

    def _get_lstm_emissions(self, sentences: torch.Tensor) -> torch.Tensor:
        """Computes emission scores from BiLSTM for batch of sentences."""
        embeds = self.dropout(self.embedding(sentences))
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        return emissions

    def _forward_alg(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes log partition function log(Z(x)) via the Forward Algorithm.
        emissions: (batch_size, seq_len, num_tags)
        mask: (batch_size, seq_len) binary float mask
        """
        batch_size, seq_len, num_tags = emissions.shape
        init_alphas = torch.full((batch_size, num_tags), -10000.0, device=emissions.device)
        init_alphas[:, self.start_idx] = 0.0
        forward_var = init_alphas

        for i in range(seq_len):
            emit_score = emissions[:, i, :].unsqueeze(2)        # (batch, num_tags, 1)
            trans_score = self.transitions.unsqueeze(0)         # (1, num_tags, num_tags)
            prev_score = forward_var.unsqueeze(1)               # (batch, 1, num_tags)
            
            # Broadcasted sum: prev_score + transitions(curr, prev) + emit_score(curr)
            # prev_score: (B, 1, K), trans_score: (1, K_to, K_from), emit_score: (B, K_to, 1)
            next_tag_var = prev_score + trans_score + emit_score
            alpha_t = log_sum_exp(next_tag_var)                 # (batch, num_tags)
            
            m = mask[:, i].unsqueeze(1)
            forward_var = m * alpha_t + (1 - m) * forward_var

        terminal_vars = forward_var + self.transitions[self.stop_idx, :].unsqueeze(0)
        return log_sum_exp(terminal_vars)

    def _score_sentence(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the unnormalized score of the target (gold) tag sequences.
        """
        batch_size, seq_len, _ = emissions.shape
        score = torch.zeros(batch_size, device=emissions.device)
        
        # Prepend START tag
        padded_tags = torch.cat(
            [torch.full((batch_size, 1), self.start_idx, dtype=torch.long, device=tags.device), tags],
            dim=1
        )
        
        for i in range(seq_len):
            from_tag = padded_tags[:, i]
            to_tag = padded_tags[:, i + 1]
            
            trans_score = self.transitions[to_tag, from_tag]
            emit_score = emissions[torch.arange(batch_size), i, to_tag]
            
            score = score + (trans_score + emit_score) * mask[:, i]

        # Transition to STOP tag at sequence end
        last_tags = []
        for b in range(batch_size):
            length = int(mask[b].sum().item())
            last_tag = tags[b, length - 1] if length > 0 else self.start_idx
            last_tags.append(last_tag)
        last_tags = torch.tensor(last_tags, device=tags.device, dtype=torch.long)
        score = score + self.transitions[self.stop_idx, last_tags]
        return score

    def forward_loss(self, sentences: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """CRF Negative Log-Likelihood Loss."""
        emissions = self._get_lstm_emissions(sentences)
        forward_score = self._forward_alg(emissions, mask)
        gold_score = self._score_sentence(emissions, tags, mask)
        loss = torch.mean(forward_score - gold_score)
        return loss

    def decode(self, sentences: torch.Tensor, mask: torch.Tensor) -> List[List[str]]:
        """
        Viterbi decoding to extract the highest-scoring sequence of tags.
        """
        emissions = self._get_lstm_emissions(sentences)
        best_tag_sequences = []

        for b in range(sentences.size(0)):
            seq_len = int(mask[b].sum().item())
            if seq_len == 0:
                best_tag_sequences.append([])
                continue

            emit = emissions[b, :seq_len]  # (seq_len, num_tags)
            
            viterbi_vars = torch.full((1, self.num_tags), -10000.0, device=sentences.device)
            viterbi_vars[0, self.start_idx] = 0.0
            backpointers = []

            for t in range(seq_len):
                bptrs_t = []
                viterbi_vars_t = []
                for next_tag in range(self.num_tags):
                    # next_tag_var = viterbi_vars + trans(next_tag, :)
                    next_tag_var = viterbi_vars + self.transitions[next_tag, :].unsqueeze(0)
                    best_prev_tag = torch.argmax(next_tag_var).item()
                    bptrs_t.append(best_prev_tag)
                    viterbi_vars_t.append(next_tag_var[0, best_prev_tag].item())

                viterbi_vars = (torch.tensor(viterbi_vars_t, device=sentences.device) + emit[t]).view(1, -1)
                backpointers.append(bptrs_t)

            terminal_vars = viterbi_vars + self.transitions[self.stop_idx, :].unsqueeze(0)
            best_last_tag = torch.argmax(terminal_vars).item()

            # Backtracking
            best_path = [best_last_tag]
            for bptrs_t in reversed(backpointers):
                best_last_tag = bptrs_t[best_last_tag]
                best_path.append(best_last_tag)

            best_path.pop()  # Remove start_idx
            best_path.reverse()
            
            # Map tag indices back to string tags
            string_tags = [self.ix_to_tag.get(idx, "O") for idx in best_path]
            best_tag_sequences.append(string_tags)

        return best_tag_sequences

    def get_transition_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Return transition parameters as a 2D numpy matrix for visualization."""
        matrix = self.transitions.detach().cpu().numpy()
        labels = [self.ix_to_tag[i] for i in range(self.num_tags)]
        return matrix, labels

