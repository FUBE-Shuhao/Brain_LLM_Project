import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from scipy.stats import zscore


PROJECT_DIR = Path("/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/scaling_finetuning-main")
ATTN_FILE = PROJECT_DIR / "model_attention" / "llama" / "7B" / "rb_p1_layer0.npy"
WORDS_LIST = PROJECT_DIR / "Analysis" / "words_list.p"

RM_SENTENCE_IDXS = [142, 143, 147, 148, 152, 153, 154]


with open(WORDS_LIST, "rb") as f:
    words_list = list(pickle.load(f))

attn = np.load(ATTN_FILE)

# 和你的回归代码保持一致
# 原始 shape 应该是 5D，并且 shape[2] 是 head
attn = attn.swapaxes(2, 0)
x_all = np.array([np.concatenate(head_blocks, axis=0) for head_blocks in attn])
x_all = np.delete(x_all, RM_SENTENCE_IDXS, axis=1)
x_all = np.nan_to_num(zscore(x_all.astype(np.float64).flatten())).reshape(x_all.shape)

X_list = []
prev_pattern_list = []
first_word_pattern_list = []

for sid, n_word in enumerate(words_list):
    n_word = int(n_word)
    tril = np.tril_indices(n_word, k=-1)

    # X: n_head × n_edges
    x_snt = x_all[:, sid, :n_word, :n_word]
    X_list.append(x_snt[:, tril[0], tril[1]])

    # trivial 1: immediately previous word
    # lower triangle里 row = col + 1
    prev = np.zeros((n_word, n_word))
    for i in range(1, n_word):
        prev[i, i - 1] = 1
    prev_pattern_list.append(prev[tril])

    # trivial 2: first word
    # lower triangle里 col = 0
    first = np.zeros((n_word, n_word))
    first[1:, 0] = 1
    first_word_pattern_list.append(first[tril])

X = np.concatenate(X_list, axis=1).T
y_prev = np.concatenate(prev_pattern_list)
y_first = np.concatenate(first_word_pattern_list)

model_prev = RidgeCV(alphas=np.logspace(1, 3, 20)).fit(X, y_prev)
pred_prev = model_prev.predict(X)
r2_prev = r2_score(y_prev, pred_prev)

model_first = RidgeCV(alphas=np.logspace(1, 3, 20)).fit(X, y_first)
pred_first = model_first.predict(X)
r2_first = r2_score(y_first, pred_first)

print("Attention file:", ATTN_FILE)
print("X shape:", X.shape)
print("R2 predicting previous-word trivial pattern:", r2_prev)
print("R2 predicting first-word trivial pattern:", r2_first)