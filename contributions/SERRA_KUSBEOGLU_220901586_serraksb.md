GitHub : serraksb 
Name : SERRA KÜSBEOĞLU 
Student number : 220901586 
MY CONTRIBUTION : crf.py

**src/models/crf.py -- Contribution Summary**

* **Linear-Chain CRF Architecture:** Implemented a Feature-based Linear-Chain CRF model using `sklearn-crfsuite` configured with L-BFGS optimization (`c1=0.1`, `c2=0.1`) for sequence labeling tasks.
* **Feature Extraction Engine:** Developed `word2features` and `sent2features` methods to extract rich orthographic characteristics (prefix/suffix n-grams, casing, digit, punctuation checks) and POS tags for each token.
* **Context-Window Modeling:** Engineered neighborhood features that capture information from the preceding and succeeding 1-2 tokens (e.g., `prev_plus_curr`, `next_word_lower`), alongside `BOS` and `EOS` sentence boundary indicators.
* **Robust Data Sanitization:** Created the `clean_feat_val` preprocessing function to safely escape newline characters, control characters, and spaces, preventing parsing errors within the crfsuite internal dump parser.
* **Transition Matrix Extraction:** Built the `get_transition_matrix` method to explicitly extract and format the learned tag-to-tag transition weights directly from the optimized model.
* **Empirical Fallback Mechanism:** Integrated a robust failsafe within the transition matrix extraction that computes empirical transition log-odds directly from the tracked training data distribution if the standard model feature extraction fails.
