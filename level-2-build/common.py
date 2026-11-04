"""Shared data handling for the Build level framework tracks.

Both the PyTorch track and the JAX track train on the exact same
corpus slice, with the same character tokenizer, the same train and
validation split, and the same batch sampler (same NumPy seed), so the
side-by-side numbers compare frameworks, not data.
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(HERE, "corpus.txt")

# Same architecture as Level 1, Learn, so the framework ports are a
# port of a known model, not a new model.
CONFIG = dict(n_layer=2, n_head=4, n_embd=64, block_size=32)
BATCH_SIZE = 16
TOTAL_STEPS = 300
RESTART_AT = 150
LEARNING_RATE = 5e-3
SEED = 1234


def load_corpus():
    with open(CORPUS_PATH, encoding="utf-8") as handle:
        return handle.read()


class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def split_data(text, tokenizer, val_fraction=0.1):
    ids = np.array(tokenizer.encode(text), dtype=np.int64)
    cut = int(len(ids) * (1.0 - val_fraction))
    return ids[:cut], ids[cut:]


def get_batch(data, rng, batch_size=BATCH_SIZE, block_size=CONFIG["block_size"]):
    starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[s:s + block_size] for s in starts])
    y = np.stack([data[s + 1:s + block_size + 1] for s in starts])
    return x, y


def mean_last(values, count=20):
    tail = values[-count:]
    return float(sum(tail) / len(tail))
