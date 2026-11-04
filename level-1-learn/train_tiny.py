"""Train the tiny GPT on a small built-in corpus and print the record.

Run: python level-1-learn/train_tiny.py
Deterministic: every random source is seeded, so the numbers in the
level README reproduce exactly on any machine with NumPy.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpt_numpy import Adam, CharTokenizer, TinyGPT, generate, get_batch, loss_and_grads

# A short public-domain style teaching corpus, repeated to give the
# model enough signal to learn from at this size. Small on purpose:
# the point at Learn level is a loss curve a reader can reproduce
# in seconds and a sample they can inspect, not a capable model.
CORPUS = (
    "to be or not to be that is the question "
    "whether tis nobler in the mind to suffer "
    "the slings and arrows of outrageous fortune "
    "or to take arms against a sea of troubles "
) * 6

CFG = dict(block_size=32, n_embd=64, n_head=4, n_layer=2)
STEPS = 300
BATCH = 16
LR = 5e-3


def main():
    tok = CharTokenizer(CORPUS)
    cfg = dict(CFG, vocab_size=tok.vocab_size)
    data = np.array(tok.encode(CORPUS), dtype=np.int64)
    model = TinyGPT(cfg, seed=7)
    opt = Adam(model.params, lr=LR)
    rng = np.random.default_rng(42)

    n_params = sum(v.size for v in model.params.values())
    print(f"vocab {tok.vocab_size} chars, params {n_params}, steps {STEPS}")

    prompt = tok.encode("to be or ")
    before = tok.decode(generate(model, prompt, 40, greedy=True))
    print(f"sample before training (greedy): {before!r}")

    losses = []
    for step in range(1, STEPS + 1):
        x, y = get_batch(data, cfg["block_size"], BATCH, rng)
        loss, grads = loss_and_grads(model, x, y)
        opt.step(model.params, grads)
        losses.append(loss)
        if step in (1, 50, 100, 150, 200, 250, STEPS):
            print(f"step {step}: loss {loss:.4f}")

    after = tok.decode(generate(model, prompt, 40, greedy=True))
    print(f"sample after training (greedy): {after!r}")
    sampled = tok.decode(generate(
        model, prompt, 40, temperature=0.8, top_k=10,
        rng=np.random.default_rng(3)))
    print(f"sample after training (temp 0.8, top-k 10): {sampled!r}")
    print(f"first-step loss {losses[0]:.4f}, final loss {losses[-1]:.4f}, "
          f"mean of last 20 steps {float(np.mean(losses[-20:])):.4f}")


if __name__ == "__main__":
    main()
