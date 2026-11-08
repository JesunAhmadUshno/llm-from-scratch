# BUILD-SPEC: llm-from-scratch Level 2

Scope, from PROPOSAL.md: train the same small GPT in PyTorch (v1 core)
and in JAX (added track) on the same public text set, with
checkpointing and one restart mid-run per track, and publish
side-by-side numbers: tokens per second, memory, final loss.

Decisions taken at Build open (2026-11-04):
- Corpus: Hamlet, Project Gutenberg ebook 1524, public domain. The
  committed corpus.txt is the first 20,000 characters starting at
  ACT I, so the repo is self-contained and the run reproduces offline.
- Architecture: identical to Level 1 (2 layers, 4 heads, embedding
  64, block 32), identical training steps (300), batch (16), learning
  rate (5e-3), and data sampler seed for both tracks.
- JAX track uses core JAX (explicit pytree, jax.grad, jax.jit,
  hand-written Adam). Flax, Optax, and Orbax are not available in
  this environment; the checkpoint is a pickle of arrays. Named as a
  deduction, carried to the ROADMAP v1.1 backlog at Deep.
- Memory is reported as one process peak RSS for the combined run,
  because the two tracks run sequentially in one process. Per-track
  memory is not claimed.

Acceptance: both tracks train end to end, each restarts from its own
checkpoint at step 150 and finishes at step 300, held-out validation
loss falls on both tracks, tests pass, CI is green, and the README
table carries only measured values from the recorded run.
