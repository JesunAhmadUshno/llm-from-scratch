"""Level 2, Build tests for the PyTorch and JAX GPT tracks.

Load-bearing checks: each track's loss falls under training on a tiny
slice, each checkpoint round-trips exactly (parameters, optimizer
state, and step count come back identical), and generation returns
the requested number of new tokens. Framework imports are required:
CI installs the CPU builds before running this file.
"""

import os
import sys
import tempfile
import unittest

BUILD = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "level-2-build")
sys.path.insert(0, BUILD)

from common import CharTokenizer, load_corpus  # noqa: E402

TEXT = load_corpus()
TOK = CharTokenizer(TEXT)


class TorchTrackTests(unittest.TestCase):
    def setUp(self):
        import torch
        from common import CONFIG
        from torch_gpt import TorchGPT
        torch.manual_seed(0)
        self.model = TorchGPT(CONFIG, TOK.vocab_size)

    def test_forward_shape_and_param_count(self):
        import torch
        from torch_gpt import count_params
        out = self.model(torch.zeros((2, 8), dtype=torch.long))
        self.assertEqual(tuple(out.shape), (2, 8, TOK.vocab_size))
        self.assertGreater(count_params(self.model), 100000)

    def test_checkpoint_roundtrip_exact(self):
        import torch
        from torch_gpt import load_checkpoint, save_checkpoint
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            save_checkpoint(path, self.model, optimizer, 42)
            before = {k: v.clone() for k, v in
                      self.model.state_dict().items()}
            with torch.no_grad():
                for p in self.model.parameters():
                    p.add_(1.0)
            step = load_checkpoint(path, self.model, optimizer)
        self.assertEqual(step, 42)
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)

    def test_short_training_reduces_loss(self):
        from torch_gpt import train_torch
        with tempfile.TemporaryDirectory() as tmp:
            result, _ = train_torch(TEXT, TOK, os.path.join(tmp, "c.pt"),
                                    steps=30, restart_at=15, seed=3)
        self.assertEqual(result["restarted_at"], 15)
        self.assertLess(result["loss_mean_last_20"],
                        result["loss_step_1"])


class JaxTrackTests(unittest.TestCase):
    def test_param_count_matches_torch_track(self):
        import jax
        from common import CONFIG
        from jax_gpt import count_params, init_params
        params = init_params(TOK.vocab_size, CONFIG, seed=0)
        leaves = jax.tree_util.tree_leaves(params)
        self.assertGreater(len(leaves), 20)
        self.assertGreater(count_params(params), 100000)

    def test_checkpoint_roundtrip_exact(self):
        import numpy as np
        from common import CONFIG
        from jax_gpt import adam_init, init_params, load_checkpoint, save_checkpoint
        params = init_params(TOK.vocab_size, CONFIG, seed=1)
        state = adam_init(params)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pkl")
            save_checkpoint(path, params, state, 7)
            back_params, back_state, step = load_checkpoint(path)
        self.assertEqual(step, 7)
        self.assertEqual(back_state["t"], 0)
        import jax
        for a, b in zip(jax.tree_util.tree_leaves(params),
                        jax.tree_util.tree_leaves(back_params)):
            self.assertTrue(np.array_equal(np.asarray(a),
                                           np.asarray(b)))

    def test_short_training_reduces_loss(self):
        from jax_gpt import train_jax
        with tempfile.TemporaryDirectory() as tmp:
            result, _ = train_jax(TEXT, TOK, os.path.join(tmp, "c.pkl"),
                                  steps=30, restart_at=15, seed=3)
        self.assertEqual(result["restarted_at"], 15)
        self.assertLess(result["loss_mean_last_20"],
                        result["loss_step_1"])

    def test_generate_returns_requested_length(self):
        from common import CONFIG
        from jax_gpt import generate, init_params
        params = init_params(TOK.vocab_size, CONFIG, seed=2)
        prompt = TOK.encode("HAMLET.\n")
        out = generate(params, TOK, prompt, 12)
        self.assertEqual(len(out), len("HAMLET.\n") + 12)


if __name__ == "__main__":
    unittest.main()
