GitHub : elifkylpp
Name : ELİF KAYAALP
Student number : 220901610
MY CONTRİBUTİON : bilstm_crf.py

### `src/models/bilstm_crf.py` -- Contribution Summary

* **BiLSTM-CRF Architecture:** Implemented a PyTorch BiLSTM-CRF model from scratch, combining a bidirectional LSTM (`bidirectional=True`) for contextual embeddings with a Linear-Chain Conditional Random Field (CRF) layer to model tag-to-tag transition probabilities.
* **Forward Algorithm (Loss Computation):** Built the `_forward_alg` method using dynamic programming to calculate the log partition function $\log(Z(x))$ and compute the CRF Negative Log-Likelihood (NLL) loss during training.
* **Viterbi Decoding:** Implemented standard Viterbi decoding to extract the globally optimal sequence of IOB2 tags across the 11-class space during inference.
* **Hardcoded Structural Transition Mask (-10000.0 Shield):** Created an immutable `illegal_mask` to physically block syntactically invalid BIO transitions (direct `O` to `I-X` jumps, transitioning into `<START>`, or leaving `<STOP>`) by locking their transition scores to `-10000.0`.
* **O-Tag Margin Adjustment (`o_margin`):** Added an `o_margin=0.90` penalty inside the `decode` function to reduce majority-class bias toward the neutral `O` tag and boost recall on toxic spans.
* **Auxiliary Weighted Loss:** Integrated smoothed inverse-frequency class weighting (`0.3 * aux_loss`) alongside the main CRF loss to prevent minority toxic categories (`THREAT`, `IDENTITY_ATTACK`) from being overwhelmed during optimization.
