# Tutorial: P01 llm-from-scratch, course module

Module: P01-llm-from-scratch
By Jesun Ahmad Ushno
Date: 2026-11-04
License for this tutorial: CC BY 4.0. Code in the repo: Apache-2.0.

## What you will do

You will clone the llm-from-scratch repo, run a tiny GPT written in
pure NumPy with hand-written backpropagation, port-check the same
model in PyTorch and JAX, and run the KV cache study that closes the
Deep level. Every command below exists in the repo as shipped at
v1.0. Learn needs NumPy only. Build needs PyTorch (CPU) and JAX.
Deep needs PyTorch (CPU). No API and no credential are needed.

All recorded numbers in this module come from the runs recorded in
the level READMEs on 2026-11-04, Muse VM, CPU only. They prove the
machinery, the framework comparison method, and the cache mechanism.
They do not prove language quality.

## Step 0: Clone and enter the repo

```text
git clone https://github.com/JesunAhmadUshno/llm-from-scratch.git
cd llm-from-scratch
```

Optional smoke check, before the levels:

```text
python tests/test_smoke.py
```

Expected: a line that says the scaffold smoke test passed.

## Step 1: Level 1, Learn, the NumPy GPT

Read level-1-learn/README.md first. It explains the model
(tokenizer, causal attention, transformer blocks, language model
head), the hand-written backward pass, and why the gradient check,
not the loss curve, is the proof.

Install the one dependency and run the Learn tests:

```text
pip install numpy
python -m unittest discover -s tests -p "test_learn*.py" -v
```

Expected: 8 tests, all passing, including the gradient check
(analytic gradients against central finite differences, agreement
below 1e-4 relative error) and the causality test.

Run the training script:

```text
python level-1-learn/train_tiny.py
```

Read your results:

- Config: 2 layers, 4 heads, embedding 64, 104,704 parameters, 300
  Adam steps on a short repeated Hamlet passage.
- Expected loss: 3.2137 at step 1 to 0.1330 at step 300, mean of the
  last 20 steps 0.1304. The run is seeded end to end, and a second
  run on 2026-11-04 reproduced the output identically.
- The run takes about a minute on the recording machine. On a slower
  shared CPU it can take several minutes.
- Read the sample honestly: the model memorised the tiny passage.
  That is the point of this level.

## Step 2: Level 2, Build, the same model in PyTorch and JAX

Read level-2-build/README.md and level-2-build/BUILD-SPEC.md. Both
tracks train on the same 20,000-character Hamlet slice (Project
Gutenberg 1524, committed as corpus.txt), share one tokenizer and
split, and restart from a checkpoint at step 150 of 300.

Install the frameworks (CPU builds):

```text
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install jax
```

Run the Build tests:

```text
python -m unittest discover -s tests -p "test_build*.py" -v
```

Expected: 7 Build tests, all passing, including exact checkpoint
round-trips on both tracks.

Run the side-by-side build:

```text
python level-2-build/run_build.py
```

Read your results:

- Parameters: 110,916 on each track.
- Expected validation loss: PyTorch 4.3464 before training to 2.0351
  after; JAX 4.2430 to 2.4048.
- Expected training speed on the recording machine: 7,716.7 tokens
  per second (PyTorch) and 11,962.7 (JAX). Your absolute numbers
  will differ by machine; the method is the comparison.
- Expected combined peak memory on the recording run: 1,066.4 MB
  RSS for both tracks in one process.
- One named gap: the JAX track is core JAX only. No Flax, no Optax,
  no Orbax. That library layer is v1.1 backlog in REPORT.md (Future work).

## Step 3: Level 3, Deep, the KV cache study

Read level-3-deep/README.md and level-3-deep/DEEP-SPEC.md. This is a
compressed Deep: the KV cache and the hand-written against fused
attention study, plus the written PyTorch vs JAX comparison, the
debugging log, and the TPU note. The rest of the v2 Deep scope is
the v1.1 backlog in REPORT.md (Future work) at the repo root.

Run the Deep tests:

```text
python -m unittest discover -s tests -p "test_deep*.py" -v
```

Expected: 9 Deep tests, all passing, including cache growth by
exactly 1 per step, hand-written against SDPA agreement inside
1e-5, and cache causality.

Run the Deep benchmark:

```text
python level-3-deep/run_deep.py
```

The runner writes level-3-deep/results_deep.json. Read your
results against the recorded run:

- Method: greedy generation of 64 tokens from the 14-token prompt
  "HAMLET." plus a newline plus "To be,", mean of 5 generations,
  seeded untrained weights.
- Config A (2 layers, embedding 64, 117,060 parameters): recorded
  284.4 tokens per second naive, 664.0 cached.
- Config B (4 layers, embedding 128, 827,204 parameters): recorded
  78.6 naive, 290.4 cached.
- Correctness: cached output was token-identical to naive output
  over 32 checked tokens on both configs.
- Prefill (batch 4, sequence 128, mean of 50 passes): recorded
  hand-written 0.0105 seconds against SDPA 0.0183 on Config A, and
  0.0424 against 0.1001 on Config B. In the first run of the same
  runner SDPA was faster at Config B, so no stable prefill winner
  is claimed. Expect your winner to vary too.
- Absolute CPU times move a lot between runs on a shared machine.
  The generation ordering (cached faster than naive) was stable;
  the absolute numbers were not.

## Step 4: Run the full test suite

```text
python -m unittest discover -s tests -v
```

Expected: 24 tests in total (8 Learn, 7 Build, 9 Deep), all
passing.

## Step 5: Check your understanding

1. Why does a falling loss not prove the hand-written backward pass
   is correct, and what check in this repo does prove it? (Answer:
   a wrong gradient can still descend on a tiny corpus; the
   finite-difference gradient check below 1e-4 on every parameter
   tensor is the proof.)
2. The Build JAX track is faster in tokens per second but ends with
   a higher validation loss than PyTorch. Is JAX the worse learner
   here? (Answer: no claim is made. Initialisation and batch order
   are seeded per track, not bit-identical across frameworks, so the
   loss gap is a recipe difference, not a framework quality result.)
3. Cached generation was 2.34 times faster on Config A and 3.69
   times on Config B in the recorded run, not the textbook quadratic
   win. Why? (Answer: at these tiny sizes per-call overhead is a
   large share of both modes, and the cached mode still runs
   embeddings, the MLP, and the head for every token.)

## Limits to keep with the numbers

- Tiny models and corpora, CPU only, no GPU or TPU number anywhere.
- Deep weights are seeded and untrained; the study measures compute.
- The JAX track is core JAX only (no Flax, Optax, or Orbax).
- Compressed Deep only; the rest of the Deep scope is REPORT.md (Future work).
