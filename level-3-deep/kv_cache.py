"""Level 3, Deep: hand-written attention, a KV cache, and the fused
comparison path, in PyTorch.

One small GPT (same block shape as Levels 1 and 2: token and
position embeddings, pre-norm blocks, GELU MLP, language model
head), but the attention is written out by hand:

- hand_attention: scores are Q @ K^T / sqrt(head_dim), an explicit
  causal mask is added, softmax normalises, and the output is the
  weighted sum over V. Nothing is fused and nothing is hidden.
- sdpa_attention: the same Q, K, V through
  torch.nn.functional.scaled_dot_product_attention, the fused
  library path, used only as the comparison in the prefill
  benchmark and in the closeness test.

Generation runs in two modes on the same weights:

- Naive: every step recomputes the full window, so a step at
  context length T costs attention over T positions for every
  token in the window.
- KV cached: a prefill pass over the prompt fills a per layer
  cache of K and V. Each later step computes Q, K, V for the one
  new token only, appends its K and V to the cache (the cache
  sequence length grows by exactly 1 per step), and attends over
  the cache. The MLP and embeddings still run per token; only the
  attention history is reused.

Fail closed on context: prompt plus generated tokens must fit
block_size. There is no eviction policy, because a sliding cache
would silently diverge from the naive mode this level compares
against, so generate raises ValueError past the bound instead.
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# Benchmark configs. Both keep the prompt (14 tokens) plus
# generated tokens inside block_size, per the fail closed rule.
CONFIG_A = dict(n_layer=2, n_head=4, n_embd=64, block_size=128)
CONFIG_B = dict(n_layer=4, n_head=4, n_embd=128, block_size=128)


def hand_attention(q, k, v, causal=True):
    """Explicit causal attention. q, k, v: (batch, heads, seq, dim).

    When the query length is 1 (a cached generation step), the one
    query sits at the last cached position, so every cached key is
    in its past and no mask applies; causal masking for that case
    would also need a mask shaped (1, seq_k), not the square mask.
    """
    head_dim = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
    if causal and q.shape[-2] > 1:
        seq_q, seq_k = scores.shape[-2], scores.shape[-1]
        mask = torch.triu(
            torch.full((seq_q, seq_k), float("-inf"),
                       dtype=scores.dtype, device=scores.device),
            diagonal=1)
        scores = scores + mask
    weights = torch.softmax(scores, dim=-1)
    return weights @ v


def sdpa_attention(q, k, v, causal=True):
    """The fused library path on the same tensors.

    is_causal=True is only valid when query and key lengths match
    (prefill). A single cached query attends to all cached keys, so
    the step case passes causal=False explicitly.
    """
    return F.scaled_dot_product_attention(
        q, k, v, is_causal=bool(causal and q.shape[-2] == k.shape[-2]))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg["n_head"]
        self.n_embd = cfg["n_embd"]
        self.head_dim = cfg["n_embd"] // cfg["n_head"]
        self.ln1 = nn.LayerNorm(cfg["n_embd"])
        self.wq = nn.Linear(cfg["n_embd"], cfg["n_embd"])
        self.wk = nn.Linear(cfg["n_embd"], cfg["n_embd"])
        self.wv = nn.Linear(cfg["n_embd"], cfg["n_embd"])
        self.wo = nn.Linear(cfg["n_embd"], cfg["n_embd"])
        self.ln2 = nn.LayerNorm(cfg["n_embd"])
        self.mlp = nn.Sequential(
            nn.Linear(cfg["n_embd"], 4 * cfg["n_embd"]), nn.GELU(),
            nn.Linear(4 * cfg["n_embd"], cfg["n_embd"]))

    def _heads(self, x):
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.n_head,
                      self.head_dim).transpose(1, 2)

    def _qkv(self, h):
        return self._heads(self.wq(h)), self._heads(self.wk(h)), \
            self._heads(self.wv(h))

    def forward_full(self, x, use_sdpa=False):
        h = self.ln1(x)
        q, k, v = self._qkv(h)
        attn = sdpa_attention(q, k, v) if use_sdpa else \
            hand_attention(q, k, v)
        out = attn.transpose(1, 2).reshape(x.shape[0], x.shape[1],
                                           self.n_embd)
        x = x + self.wo(out)
        x = x + self.mlp(self.ln2(x))
        return x

    def prefill(self, x):
        """Full pass that also returns this layer's K and V cache."""
        h = self.ln1(x)
        q, k, v = self._qkv(h)
        attn = hand_attention(q, k, v)
        out = attn.transpose(1, 2).reshape(x.shape[0], x.shape[1],
                                           self.n_embd)
        x = x + self.wo(out)
        x = x + self.mlp(self.ln2(x))
        return x, {"k": k, "v": v}

    def forward_step(self, x, cache):
        """One new token. x: (batch, 1, n_embd). Appends to cache."""
        h = self.ln1(x)
        q, k_new, v_new = self._qkv(h)
        k = torch.cat([cache["k"], k_new], dim=2)
        v = torch.cat([cache["v"], v_new], dim=2)
        attn = hand_attention(q, k, v, causal=False)
        out = attn.transpose(1, 2).reshape(x.shape[0], 1, self.n_embd)
        x = x + self.wo(out)
        x = x + self.mlp(self.ln2(x))
        return x, {"k": k, "v": v}


class DeepGPT(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = dict(cfg)
        self.tok_emb = nn.Embedding(vocab_size, cfg["n_embd"])
        self.pos_emb = nn.Embedding(cfg["block_size"], cfg["n_embd"])
        self.blocks = nn.ModuleList(
            [Block(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["n_embd"])
        self.head = nn.Linear(cfg["n_embd"], vocab_size)

    def forward_full(self, idx, use_sdpa=False):
        pos = torch.arange(idx.shape[1], device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block.forward_full(x, use_sdpa=use_sdpa)
        return self.head(self.ln_f(x))

    def prefill(self, idx):
        """Logits for every prompt position, plus the filled caches."""
        pos = torch.arange(idx.shape[1], device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        caches = []
        for block in self.blocks:
            x, cache = block.prefill(x)
            caches.append(cache)
        return self.head(self.ln_f(x)), caches

    def step(self, token, position, caches):
        """Logits for one new token at its absolute position."""
        x = self.tok_emb(token) + self.pos_emb(
            torch.tensor([position], device=token.device))
        x = x.unsqueeze(1) if x.dim() == 2 else x
        new_caches = []
        for block, cache in zip(self.blocks, caches):
            x, new_cache = block.forward_step(x, cache)
            new_caches.append(new_cache)
        return self.head(self.ln_f(x)), new_caches

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def _check_context(cfg, total):
    if total > cfg["block_size"]:
        raise ValueError(
            f"context length {total} exceeds block_size "
            f"{cfg['block_size']}; this cache has no eviction policy")


@torch.no_grad()
def generate_naive(model, prompt_ids, max_new):
    """Greedy generation, recomputing the full window each step."""
    model.eval()
    _check_context(model.cfg, len(prompt_ids) + max_new)
    ids = list(prompt_ids)
    for _ in range(max_new):
        idx = torch.tensor([ids], dtype=torch.long)
        logits = model.forward_full(idx)[0, -1]
        ids.append(int(torch.argmax(logits)))
    return ids


@torch.no_grad()
def generate_cached(model, prompt_ids, max_new):
    """Greedy generation with the KV cache. Same tokens, less work."""
    model.eval()
    _check_context(model.cfg, len(prompt_ids) + max_new)
    ids = list(prompt_ids)
    idx = torch.tensor([ids], dtype=torch.long)
    logits, caches = model.prefill(idx)
    last = logits[0, -1]
    for step_i in range(max_new):
        nxt = int(torch.argmax(last))
        ids.append(nxt)
        if step_i == max_new - 1:
            break
        token = torch.tensor([[nxt]], dtype=torch.long)
        logits, caches = model.step(token, len(ids) - 1, caches)
        last = logits[0, -1]
    return ids


def cache_shapes(caches):
    """Per layer (K shape, V shape), for tests and the runner."""
    return [(tuple(c["k"].shape), tuple(c["v"].shape)) for c in caches]


def benchmark_generation(model, prompt_ids, max_new, mode, repeats=3):
    """Mean wall time and tokens per second for one generation mode.

    Tokens per second counts generated tokens only, over the mean
    of `repeats` full generations.
    """
    generate = generate_naive if mode == "naive" else generate_cached
    generate(model, prompt_ids, min(4, max_new))  # warmup, untimed
    start = time.perf_counter()
    for _ in range(repeats):
        out = generate(model, prompt_ids, max_new)
    elapsed = (time.perf_counter() - start) / repeats
    return {
        "mode": mode,
        "tokens_generated": max_new,
        "repeats": repeats,
        "seconds_mean": round(elapsed, 4),
        "tokens_per_second": round(max_new / elapsed, 1),
        "output_length": len(out),
    }


def benchmark_prefill(model, seq_len, batch_size, use_sdpa,
                      repeats=20, warmup=3):
    """Mean prefill time for the hand-written or the SDPA path."""
    torch.manual_seed(7)
    vocab = model.head.out_features
    idx = torch.randint(0, vocab, (batch_size, seq_len))
    for _ in range(warmup):
        model.forward_full(idx, use_sdpa=use_sdpa)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            model.forward_full(idx, use_sdpa=use_sdpa)
    elapsed = (time.perf_counter() - start) / repeats
    return {
        "path": "sdpa" if use_sdpa else "hand",
        "batch_size": batch_size,
        "seq_len": seq_len,
        "repeats": repeats,
        "seconds_mean": round(elapsed, 4),
        "tokens_per_second": round(batch_size * seq_len / elapsed, 1),
    }
