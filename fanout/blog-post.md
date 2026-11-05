# A Model From Nothing: Building a GPT From Scratch, Then Proving the Cache

By Jesun Ahmad Ushno
Date: 2026-11-04
Status: DRAFT, written fan-out for P01 llm-from-scratch v1.0. Not posted.

## The problem

Frameworks make a language model look easy: a few library calls, a
training loop, a falling loss. None of that proves the builder
understands attention, backpropagation, or why generation is slow.
This project, P01 in Roadmap v3, removes the hiding places in three
steps. First, a tiny GPT in pure NumPy with the backward pass written
by hand and proven against finite differences. Second, the same model
ported to PyTorch and to JAX and trained side by side on the same
text. Third, the hardest inference idea in the spec, a KV cache, built
and benchmarked against naive generation and against fused attention.

Repository: https://github.com/JesunAhmadUshno/llm-from-scratch

## Past, present, future (the dossier, condensed)

Past: nanoGPT-style teaching builds and framework tutorials already
exist. The prior art is recorded in NOVELTY.md in the repo, and the
novelty text there stays marked PROPOSED. This post makes no novelty
claim beyond the measured work below.

Present: v1.0 closes three levels on 2026-11-04. Learn proved the
machinery framework-free. Build ran the same model in two frameworks
on a real text slice with checkpoint restarts. Compressed Deep built
the single hardest Deep capability in the SPEC, the KV cache plus the
hand-written against fused attention speed study, plus the written
PyTorch vs JAX comparison, the debugging log, and the TPU note.

Future: the rest of the v2 Deep scope is named in ROADMAP.md in the
repo as the v1.1 backlog. It includes the Flax, Optax, and Orbax
library layer, a JAX KV cache, larger and longer training, GPU
benchmarks, a profiler trace for the XLA fusion claim, and a real TPU
run. None of that is done, and none of it is claimed here.

## Main results claim

At v1.0: Learn built a tiny GPT in pure NumPy with 104,704 parameters and hand-written backpropagation, training loss fell from 3.2137 to 0.1330 over 300 steps (mean of the last 20 steps 0.1304), the gradient check agreed with central finite differences below 1e-4 relative error, 8 tests passed, and a second run reproduced the output identically; Build ported the same model to PyTorch and to JAX at 110,916 parameters each, trained both on a 20,000-character Hamlet slice (Project Gutenberg 1524) with a checkpoint restart at step 150 of 300, validation loss fell from 4.3464 to 2.0351 in PyTorch and from 4.2430 to 2.4048 in JAX, training speed was 7,716.7 tokens per second in PyTorch and 11,962.7 in JAX on CPU, combined peak memory was 1,066.4 MB RSS, and 7 Build tests passed; compressed Deep added a KV cache study in PyTorch, greedy generation of 64 tokens from a 14-token prompt as a mean of 5 generations, Config A (117,060 parameters) ran at 284.4 tokens per second naive and 664.0 cached, Config B (827,204 parameters) ran at 78.6 naive and 290.4 cached, cached output was token-identical to naive output over 32 tokens on both configs, the hand-written versus SDPA prefill comparison had no stable winner (the winner flipped between runs), and 9 Deep tests passed, for 24 tests in total.

Every number in that claim comes from a run recorded in the level
READMEs in the repo, on the Muse VM, CPU only. Read the limits
section below before quoting any of it.

## Results by level

### Level 1, Learn (release v0.1)

The whole model, including backpropagation and Adam, is plain NumPy
in gpt_numpy.py. The load-bearing proof is not the loss curve: the
test suite compares analytic gradients against central finite
differences on sampled entries of every parameter tensor.

| Measure | Result |
| --- | --- |
| Parameters | 104,704 (2 layers, 4 heads, embedding 64) |
| Training loss, step 1 to step 300 | 3.2137 to 0.1330 |
| Mean loss, last 20 steps | 0.1304 |
| Gradient check | Agreement below 1e-4 relative error |
| Reproduction | Second run produced identical output |
| Tests | 8 |

Read honestly: the model memorised a tiny repeated Hamlet passage.
That is the Learn point, the machinery works and its gradients are
proven, not language ability.

### Level 2, Build (release v0.5)

The same architecture in PyTorch (autograd, torch.optim Adam,
torch.save checkpoints) and in core JAX (explicit parameter pytree,
jax.grad, a jit compiled step, hand-written Adam, pickle checkpoint).
Both tracks share one corpus, one tokenizer, one split, and one
sampler, and both restart from a checkpoint at step 150 of 300.

| Measure | PyTorch | JAX |
| --- | --- | --- |
| Parameters | 110,916 | 110,916 |
| Validation loss, before to after | 4.3464 to 2.0351 | 4.2430 to 2.4048 |
| Training tokens per second, CPU | 7,716.7 | 11,962.7 |
| Checkpoint restart | Step 150, resumed to 300 | Step 150, resumed to 300 |

Combined peak memory for the two tracks in one process: 1,066.4 MB
RSS. Per-track memory is not claimed. Tests: 7 Build tests, 15 total
with Learn. At 300 steps on 18,000 training characters both models
learn the play's shape and neither writes English, as the Build
README states plainly.

### Level 3, Deep, compressed (release v1.0)

A PyTorch GPT with hand-written attention and two generation modes on
the same seeded untrained weights: naive recomputation, and a KV
cache whose sequence length grows by exactly 1 per step, test
enforced. The study measures compute, not language quality.

| Config | Params | Naive tok/s | Cached tok/s |
| --- | --- | --- | --- |
| A (2 layers, embedding 64) | 117,060 | 284.4 | 664.0 |
| B (4 layers, embedding 128) | 827,204 | 78.6 | 290.4 |

Method: greedy generation of 64 tokens from a 14-token prompt, mean
of 5 generations. Cached output was token-identical to naive output
over 32 checked tokens on both configs. Prefill at batch 4, sequence
128 took 0.0105 seconds hand-written against 0.0183 seconds SDPA on
Config A, and 0.0424 against 0.1001 on Config B, in the recorded run.
In the first run of the same runner, SDPA was faster at Config B, so
the prefill winner flips between runs at these shapes on this CPU and
no stable prefill winner is claimed. Tests: 9 Deep tests, 24 total.

## Honest limits

- Tiny models, tiny corpora, CPU only. Learn memorised a repeated
  passage. Build trained 300 steps on 18,000 characters. Deep used
  seeded untrained weights. No result here is a language quality
  result.
- The JAX track uses core JAX only: no Flax model definition, no
  Optax optimizer, no Orbax checkpoint. SPEC.md names those
  libraries. The mechanisms are real and tested; the named library
  layer is the v1.1 backlog.
- One configuration per level, one corpus per level, runs on the
  Muse VM, not the assigned BlackBox GPU node. No GPU number and no
  TPU number is claimed. The TPU section in the Deep README is
  reasoning only.
- Absolute CPU timings moved between runs on this shared VM; the
  cache speed ratios ranged from 2.10 to 5.36 times across runs and
  configs and the prefill winner flipped. The recorded tables are one
  run's measurement with method, not constants of nature.
- Compressed Deep is not the full v2 Deep scope. The remainder is the
  named v1.1 backlog in ROADMAP.md.
- Two level READMEs carry stale status lines from their own ship
  day (Learn says Build and Deep are not started; Build says Deep is
  not started). The root README and the later level READMEs supersede
  those lines. The numbers in this post follow the root README where
  any conflict appears.

## Where to go next

- Tutorial: fanout/tutorial.md in the repo, a step-by-step module
  that runs all three levels and their test suites.
- Releases: v0.1 (Learn), v0.5 (Build), v1.0 (Learn, Build,
  compressed Deep) on the GitHub repo linked above.
- Book chapter: "A Model From Nothing" is the named chapter in
  SPEC.md; the draft status is logged in PUBLICITY.md in the repo.
- License: code Apache-2.0, tutorials CC BY 4.0.
