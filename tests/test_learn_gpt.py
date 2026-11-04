"""Level 1, Learn tests for the NumPy GPT.

The load-bearing test is the gradient check: analytic gradients from
the hand-written backward pass must match central finite differences.
If backprop were wrong, training would still run and print a loss, so
only this check can prove it right.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "level-1-learn"),
)
from gpt_numpy import (Adam, CharTokenizer, TinyGPT, generate, get_batch,
                       loss_and_grads)

CORPUS = "to be or not to be that is the question "


def tiny_model(seed=1):
    tok = CharTokenizer(CORPUS)
    cfg = dict(vocab_size=tok.vocab_size, block_size=6, n_embd=12,
               n_head=3, n_layer=1)
    return tok, TinyGPT(cfg, seed=seed)


def tiny_batch(tok):
    data = np.array(tok.encode(CORPUS * 2))
    x = np.stack([data[0:6], data[3:9]])
    y = np.stack([data[1:7], data[4:10]])
    return x, y


class TokenizerTests(unittest.TestCase):
    def test_roundtrip(self):
        tok, _ = tiny_model()
        self.assertEqual(tok.decode(tok.encode(CORPUS)), CORPUS)

    def test_vocab_covers_corpus(self):
        tok, _ = tiny_model()
        self.assertEqual(tok.vocab_size, len(set(CORPUS)))


class ForwardTests(unittest.TestCase):
    def test_logits_shape(self):
        tok, model = tiny_model()
        x, _ = tiny_batch(tok)
        logits, _ = model.forward(x)
        self.assertEqual(logits.shape, (2, 6, tok.vocab_size))

    def test_causality_future_token_cannot_change_past_logits(self):
        tok, model = tiny_model()
        x, _ = tiny_batch(tok)
        logits_a, _ = model.forward(x)
        x2 = x.copy()
        x2[:, -1] = (x2[:, -1] + 1) % tok.vocab_size
        logits_b, _ = model.forward(x2)
        np.testing.assert_allclose(
            logits_a[:, :-1], logits_b[:, :-1], rtol=0, atol=1e-10)


class GradientCheckTests(unittest.TestCase):
    def test_analytic_gradients_match_finite_differences(self):
        tok, model = tiny_model()
        x, y = tiny_batch(tok)
        _, grads = loss_and_grads(model, x, y)
        rng = np.random.default_rng(0)
        eps = 1e-5
        checked = 0
        for key, arr in model.params.items():
            for _ in range(3):
                idx = tuple(rng.integers(0, s) for s in arr.shape)
                orig = arr[idx]
                arr[idx] = orig + eps
                plus = model.loss(x, y)
                arr[idx] = orig - eps
                minus = model.loss(x, y)
                arr[idx] = orig
                numeric = (plus - minus) / (2 * eps)
                analytic = grads[key][idx]
                rel = abs(numeric - analytic) / max(
                    1e-8, abs(numeric) + abs(analytic))
                self.assertLess(
                    rel, 1e-4,
                    f"gradient mismatch at {key}{idx}: numeric "
                    f"{numeric:.8f}, analytic {analytic:.8f}")
                checked += 1
        self.assertGreaterEqual(checked, 30)


class TrainingTests(unittest.TestCase):
    def test_loss_decreases_on_tiny_corpus(self):
        tok = CharTokenizer(CORPUS * 6)
        cfg = dict(vocab_size=tok.vocab_size, block_size=16, n_embd=32,
                   n_head=4, n_layer=2)
        model = TinyGPT(cfg, seed=7)
        opt = Adam(model.params, lr=5e-3)
        rng = np.random.default_rng(42)
        data = np.array(tok.encode(CORPUS * 6), dtype=np.int64)
        losses = []
        for _ in range(60):
            x, y = get_batch(data, cfg["block_size"], 8, rng)
            loss, grads = loss_and_grads(model, x, y)
            opt.step(model.params, grads)
            losses.append(loss)
        self.assertLess(float(np.mean(losses[-10:])), losses[0])


class SamplingTests(unittest.TestCase):
    def test_greedy_is_deterministic(self):
        _, model = tiny_model()
        prompt = [0, 1, 2]
        a = generate(model, prompt, 8, greedy=True)
        b = generate(model, prompt, 8, greedy=True)
        self.assertEqual(a, b)
        self.assertEqual(len(a), len(prompt) + 8)

    def test_temperature_sampling_runs_and_respects_length(self):
        _, model = tiny_model()
        out = generate(model, [0, 1], 6, temperature=0.8, top_k=5,
                       rng=np.random.default_rng(3))
        self.assertEqual(len(out), 8)


if __name__ == "__main__":
    unittest.main()
