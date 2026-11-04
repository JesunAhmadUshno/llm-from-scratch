# DEEP-SPEC: llm-from-scratch Level 3 (compressed Deep)

Scope, from SPEC.md: Deep adds a KV cache and a hand-written against
fused attention speed study (the v1 core Deep). The added track asks
for a written PyTorch vs JAX comparison on the same model (speed,
code size, debugging pain, where XLA fusion helps), a debugging log
of tracer and shape errors and how each was fixed, and a TPU note
(when TPUs beat GPUs, what changes in code, register item 225).

Compressed Deep decision (December Compression Protocol, 2026-11-04):
the single hardest capability is the KV cache with the hand-written
against fused attention speed study, built, tested, and benchmarked
in full in PyTorch. The comparison write-up, the debugging log, and
the TPU note ship as written sections of the level README. The rest
of the v2 Deep scope goes to ROADMAP.md at the repo root as the
v1.1 backlog: the Flax, Optax, and Orbax library layer for the JAX
track, a JAX KV cache, larger and longer training, GPU benchmarks,
and a real TPU run.

Decisions taken at Deep open (2026-11-04):
- One PyTorch model (kv_cache.py) with hand-written causal attention:
  explicit QK^T scores, an explicit causal mask, softmax, and the
  weighted sum over V. No library attention in the hand path.
- Two generation modes on the same weights: naive (recompute the
  full window at every step) and KV cached (cache K and V per layer
  at prefill, then compute only the new token per step).
- A fused comparison path on the same weights and shapes:
  torch.nn.functional.scaled_dot_product_attention, benchmarked in
  prefill, where the full sequence is processed at once.
- Context bound is fail closed: prompt plus generated tokens must
  fit block_size. The cache has no eviction policy, so generation
  past block_size raises instead of silently sliding and diverging
  from the naive mode. Benchmark configs stay inside the bound.
- Weights are seeded and untrained. The benchmark measures compute,
  not language quality, and greedy equivalence between the two
  generation modes is an exact token for token check on those
  weights. Language quality was Build level work.
- Benchmark machine: this Muse VM, CPU only. No GPU number and no
  TPU number is claimed anywhere in this level.

Acceptance: greedy cached output is token-identical to greedy naive
output (test enforced), cache shapes grow by exactly 1 per step,
hand-written attention matches the SDPA path on the same inputs
inside the stated tolerance, the cache path cannot attend to future
tokens, benchmark helpers run under test, the runner writes
results_deep.json with measured values only, tests pass, CI is
green, and the README carries the measured table, the failure
modes, the comparison write-up, the debugging log, and the TPU note.
