"""Level 3, Deep tests for the KV cache and attention study.

Load-bearing checks: greedy cached generation is token-identical
to greedy naive generation on the same weights, the cache grows
by exactly 1 in sequence length per step, hand-written attention
matches the SDPA path on the same inputs inside the stated
tolerance (atol 1e-5, rtol 1e-5; the measured gap on this machine
was about 2.4e-07), the cached path cannot use future tokens, and
the benchmark helpers run and return sane values.
"""

import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "level-3-deep"))
sys.path.insert(0, os.path.join(ROOT, "level-2-build"))

from common import CharTokenizer, load_corpus  # noqa: E402
from kv_cache import (  # noqa: E402
    CONFIG_A,
    DeepGPT,
    benchmark_generation,
    benchmark_prefill,
    cache_shapes,
    generate_cached,
    generate_naive,
    hand_attention,
    sdpa_attention,
)

TOK = CharTokenizer(load_corpus())
PROMPT = TOK.encode("HAMLET.\nTo be,")


def small_model(seed=99):
    torch.manual_seed(seed)
    cfg = dict(n_layer=2, n_head=2, n_embd=32, block_size=64)
    return DeepGPT(cfg, TOK.vocab_size)


class EquivalenceTests(unittest.TestCase):
    def test_greedy_cached_equals_naive_config_a(self):
        torch.manual_seed(1234)
        model = DeepGPT(CONFIG_A, TOK.vocab_size)
        self.assertEqual(generate_cached(model, PROMPT, 32),
                         generate_naive(model, PROMPT, 32))

    def test_greedy_cached_equals_naive_small(self):
        model = small_model()
        self.assertEqual(generate_cached(model, PROMPT, 24),
                         generate_naive(model, PROMPT, 24))

    def test_context_bound_fails_closed(self):
        model = small_model()
        with self.assertRaises(ValueError):
            generate_cached(model, PROMPT, 200)
        with self.assertRaises(ValueError):
            generate_naive(model, PROMPT, 200)


class CacheShapeTests(unittest.TestCase):
    def test_cache_grows_by_one_per_step(self):
        model = small_model()
        idx = torch.tensor([PROMPT], dtype=torch.long)
        _, caches = model.prefill(idx)
        before = cache_shapes(caches)
        self.assertEqual(len(before), model.cfg["n_layer"])
        for k_shape, v_shape in before:
            self.assertEqual(k_shape, v_shape)
            self.assertEqual(k_shape[2], len(PROMPT))
        token = torch.tensor([[PROMPT[-1]]], dtype=torch.long)
        _, caches = model.step(token, len(PROMPT), caches)
        token = torch.tensor([[PROMPT[-2]]], dtype=torch.long)
        _, caches = model.step(token, len(PROMPT) + 1, caches)
        for k_shape, v_shape in cache_shapes(caches):
            self.assertEqual(k_shape[2], len(PROMPT) + 2)
            self.assertEqual(v_shape[2], len(PROMPT) + 2)

    def test_prefill_logits_match_full_forward(self):
        model = small_model()
        idx = torch.tensor([PROMPT], dtype=torch.long)
        with torch.no_grad():
            full = model.forward_full(idx)
            prefill, _ = model.prefill(idx)
        self.assertTrue(torch.allclose(full, prefill, atol=1e-6))


class AttentionTests(unittest.TestCase):
    def test_hand_attention_close_to_sdpa(self):
        torch.manual_seed(0)
        q = torch.randn(2, 4, 16, 16)
        k = torch.randn(2, 4, 16, 16)
        v = torch.randn(2, 4, 16, 16)
        diff = (hand_attention(q, k, v)
                - sdpa_attention(q, k, v)).abs().max().item()
        self.assertLess(diff, 1e-5)

    def test_full_forward_is_causal(self):
        model = small_model()
        idx = torch.tensor([PROMPT], dtype=torch.long)
        with torch.no_grad():
            a = model.forward_full(idx)
            idx2 = idx.clone()
            idx2[0, -1] = (idx2[0, -1] + 1) % TOK.vocab_size
            b = model.forward_full(idx2)
        self.assertTrue(torch.allclose(a[:, :-1], b[:, :-1],
                                       atol=1e-6))

    def test_cached_step_uses_only_past(self):
        """Step logits after a prefix equal full forward on that
        prefix: the cache path has no way to see a future token,
        because the differing future token is never fed to it."""
        model = small_model()
        prefix = PROMPT[:8]
        idx = torch.tensor([prefix], dtype=torch.long)
        with torch.no_grad():
            full_last = model.forward_full(idx)[0, -1]
            _, caches = model.prefill(
                torch.tensor([prefix[:-1]], dtype=torch.long))
            step_logits, _ = model.step(
                torch.tensor([[prefix[-1]]], dtype=torch.long),
                len(prefix) - 1, caches)
        self.assertTrue(torch.allclose(full_last, step_logits[0, -1],
                                       atol=1e-5))


class BenchmarkHelperTests(unittest.TestCase):
    def test_benchmark_helpers_run(self):
        model = small_model()
        gen = benchmark_generation(model, PROMPT, 8, "cached",
                                   repeats=1)
        self.assertEqual(gen["output_length"], len(PROMPT) + 8)
        self.assertGreater(gen["tokens_per_second"], 0)
        pre = benchmark_prefill(model, 32, 2, use_sdpa=True,
                                repeats=2, warmup=1)
        self.assertEqual(pre["path"], "sdpa")
        self.assertGreater(pre["seconds_mean"], 0)


if __name__ == "__main__":
    unittest.main()
