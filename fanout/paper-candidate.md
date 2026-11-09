# Paper candidate brief: llm-from-scratch paper

Tracker row: NONE. No row for P01 exists in the publications
pipeline tracker at 2026-11-04. This brief is written so a row can
be created from it by the batch coordinator; the brief itself does
not create or move any tracker row.
Project: P01 llm-from-scratch
Status: CANDIDATE brief, 2026-11-04
Submission note: submission is Jesun's tap. Nothing is submitted by
this brief, and no venue has been contacted.
Paper angle (per PROPOSAL.md): the from-scratch build and the KV cache
study, with the PyTorch vs JAX same-model study as the comparison
core. Book chapter named in PROPOSAL.md: "A Model From Nothing".

## Working title

One model, three builds: a from-scratch GPT in NumPy, PyTorch, and
JAX, with a KV cache and attention study on CPU

## Abstract draft

Based only on the measured v1.0 results recorded in the repo.
Numbers below are from tiny corpora and CPU runs and are stated as
such.

Framework abstractions hide the mechanics that decide whether a
language model trains and how fast it generates. This work builds
one small GPT architecture three times and measures each build. At
Learn level, the model is written in pure NumPy (104,704
parameters) with hand-written backpropagation; training loss fell
from 3.2137 to 0.1330 over 300 steps (mean of the last 20 steps
0.1304), analytic gradients agreed with central finite differences
below 1e-4 relative error on sampled entries of every parameter
tensor, and a second seeded run reproduced the output identically,
across 8 unit tests. At Build level, the same architecture was
ported to PyTorch and to core JAX at 110,916 parameters each and
trained on the same 20,000-character Hamlet slice (Project
Gutenberg 1524) with a checkpoint restart at step 150 of 300;
validation loss fell from 4.3464 to 2.0351 in PyTorch and from
4.2430 to 2.4048 in JAX, at 7,716.7 and 11,962.7 training tokens
per second on CPU, with a combined peak memory of 1,066.4 MB RSS,
across 7 Build tests. At Deep level, a KV cache study on seeded
untrained weights generated 64 tokens greedily from a 14-token
prompt (mean of 5 generations): Config A (117,060 parameters) ran
at 284.4 tokens per second naive and 664.0 cached, Config B
(827,204 parameters) ran at 78.6 naive and 290.4 cached, cached
output was token-identical to naive output over 32 tokens on both
configs, and the hand-written versus SDPA prefill comparison had
no stable winner across runs, across 9 Deep tests and 24 tests in
total. The paper claims the build discipline, the same-model
framework comparison method, and the cache result on these
configs, not language quality or production inference performance.

## Method outline

1. One architecture, three implementations: character tokenizer,
   token plus position embeddings, causal multi-head attention,
   LayerNorm, GELU MLP, residual paths, language model head. NumPy
   with manual backpropagation, PyTorch with autograd, core JAX as
   a pure function over an explicit parameter pytree.
2. Gradient proof: analytic gradients against central finite
   differences on sampled entries of every parameter tensor, line
   at 1e-4 relative error, test enforced. Causality proven by a
   test that a future token cannot change earlier logits.
3. Same-model framework comparison: one corpus, one tokenizer, one
   90/10 split, one seeded sampler, one 300-step recipe, checkpoint
   restart at step 150 on both tracks. Tokens per second counts
   training tokens only; the JAX timing includes first-step JIT
   compilation. Initialisation and batch order are seeded per
   track, not bit-identical, and the paper states that limit where
   the losses are compared.
4. KV cache study: hand-written attention with an explicit causal
   mask; naive generation recomputes the full window, cached
   generation prefills the prompt then computes one new token per
   step, cache grows by exactly 1 per step, test enforced.
   Correctness is exact greedy token equivalence over 32 tokens.
5. Fused comparison: the same weights and shapes through
   torch.nn.functional.scaled_dot_product_attention in prefill
   (batch 4, sequence 128, mean of 50 passes), with the run-to-run
   flip reported instead of a winner picked from one run.
6. Failure modes and debugging log: cache memory growth, no
   eviction past block size, near-tie argmax flips, stale cache,
   no padding support, timing noise; the JAX tracer error, the
   SDPA causal-mask alignment fix, and the cache axis fix, each
   with its guarding test.

## Results table

Measured 2026-11-04, Muse VM, CPU only, from the level READMEs in
the repo.

| Level | Headline numbers | Tests |
| --- | --- | --- |
| Learn | 104,704 params; loss 3.2137 to 0.1330 over 300 steps; last-20 mean 0.1304; gradient check below 1e-4; second run identical | 8 |
| Build, PyTorch | 110,916 params; validation loss 4.3464 to 2.0351; 7,716.7 tok/s | 7 Build (15 total with Learn) |
| Build, JAX | 110,916 params; validation loss 4.2430 to 2.4048; 11,962.7 tok/s; combined peak RSS 1,066.4 MB | (same 7) |
| Deep, Config A | 117,060 params; naive 284.4 vs cached 664.0 tok/s; prefill hand 0.0105s vs SDPA 0.0183s | 9 Deep (24 total) |
| Deep, Config B | 827,204 params; naive 78.6 vs cached 290.4 tok/s; prefill hand 0.0424s vs SDPA 0.1001s; winner flipped between runs | (same 9) |

## Venue class note

Per PROPOSAL.md the angle is a citable comparison note. Venue class:
workshop or systems and education venue, one venue at a time, per
the tracker stage rules. No venue is named as submitted, accepted,
or contacted in this brief. Fees, if any, arise only after
acceptance and remain Jesun's call under the free-only rule.

## Limits the paper must carry

- Tiny models and corpora throughout; Learn memorised a repeated
  passage by design, Build trained 300 steps, Deep weights are
  seeded and untrained. No language quality claim.
- CPU only, one machine, shared VM timing noise measured and
  reported (cache ratios 2.10 to 5.36 times across runs).
- The JAX track is core JAX only: no Flax, Optax, or Orbax. The
  named library layer is v1.1 backlog.
- Compressed Deep only. The v1.1 backlog (JAX KV cache, GPU
  benchmarks, profiler trace, TPU run) is future work, named in
  REPORT.md (Future work) in the repo.
- The TPU note in the Deep README is reasoning only and carries no
  measured claim into the paper.

## Submission gate

Submission is Jesun's tap and nothing is submitted. This brief is
the candidate only. Tracker note: no P01 row exists in the
publications pipeline tracker at 2026-11-04, so there is no row to
update; the coordinator can create one from this brief if Jesun
wants the candidate tracked.
