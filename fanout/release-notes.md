# Release notes draft: llm-from-scratch

Status: DRAFT notes for releases that already exist, 2026-11-04.
No new release is created by this file. Release URLs verified with
gh release list on 2026-11-04.

## v0.1, Learn (released 2026-11-04)

Release: https://github.com/JesunAhmadUshno/llm-from-scratch/releases/tag/v0.1
Title on GitHub: Learn v0.1

- Tiny GPT in pure NumPy (level-1-learn/gpt_numpy.py): character
  tokenizer, causal multi-head attention, transformer blocks,
  final LayerNorm, language model head.
- Full hand-written backpropagation, cross-entropy, and a small
  Adam optimiser over the parameter dict.
- Gradient proof: analytic gradients against central finite
  differences below 1e-4 relative error, test enforced, plus a
  causality test.
- Measured run: 104,704 parameters, loss 3.2137 at step 1 to
  0.1330 at step 300, mean of last 20 steps 0.1304, second run
  reproduced the output identically.
- 8 unit tests. NumPy only.

## v0.5, Build (released 2026-11-04)

Release: https://github.com/JesunAhmadUshno/llm-from-scratch/releases/tag/v0.5
Title on GitHub: Level 2 Build: PyTorch and JAX ports

- The Level 1 model ported to PyTorch (nn.Module, autograd,
  torch.optim Adam, torch.save checkpoints) and to core JAX (pure
  function over a parameter pytree, jax.grad, jit step,
  hand-written Adam, pickle checkpoint).
- Shared data: Hamlet, Project Gutenberg 1524, first 20,000
  characters committed as corpus.txt; one tokenizer, one 90/10
  split, one seeded sampler for both tracks.
- Both tracks restart from a checkpoint at step 150 of 300.
- Measured run: 110,916 parameters each; validation loss 4.3464
  to 2.0351 (PyTorch) and 4.2430 to 2.4048 (JAX); training speed
  7,716.7 and 11,962.7 tokens per second on CPU; combined peak
  RSS 1,066.4 MB.
- 7 Build tests (15 total with Learn), including exact checkpoint
  round-trips on both tracks.
- Named gap: core JAX only, no Flax, Optax, or Orbax. That layer
  is v1.1 backlog.

## v1.0, compressed Deep (released 2026-11-04)

Release: https://github.com/JesunAhmadUshno/llm-from-scratch/releases/tag/v1.0
Title on GitHub: Deep v1.0: KV cache and attention study

- PyTorch GPT with hand-written attention and two generation
  modes: naive full recomputation, and KV cached generation
  (prefill fills the cache; each step adds exactly 1 position,
  test enforced). Past block size the generate functions raise;
  there is no silent eviction.
- KV cache benchmark, seeded untrained weights, greedy 64 tokens
  from a 14-token prompt, mean of 5: Config A (117,060 params)
  284.4 naive vs 664.0 cached tokens per second; Config B
  (827,204 params) 78.6 vs 290.4. Cached output token-identical
  to naive over 32 tokens on both configs.
- Prefill study, batch 4, sequence 128: hand-written 0.0105s vs
  SDPA 0.0183s (Config A) and 0.0424s vs 0.1001s (Config B) in
  the recorded run; the winner flipped between runs at Config B,
  so no stable prefill winner is claimed.
- Written PyTorch vs JAX comparison (speed, code size, debugging
  pain, where XLA fusion helps), a debugging log of four tracer
  and shape errors with fixes, and a reasoning-only TPU note. No
  TPU number is claimed.
- Failure modes written: cache memory growth, no eviction,
  near-tie argmax flips, stale cache, no padding, timing noise.
- 9 Deep tests (24 total with Learn and Build).
- Compressed Deep under the December Compression Protocol: the
  rest of the v2 Deep scope is the v1.1 backlog in ROADMAP.md.
