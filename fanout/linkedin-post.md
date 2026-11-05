# LinkedIn post draft: P01 llm-from-scratch v1.0

Status: DRAFT NOT POSTED
Date drafted: 2026-11-04
Author: Jesun Ahmad Ushno
Rule for this platform: the link goes in the first comment, not in the
post body. The first comment line is at the end of this file.

## Post body

I built a GPT from nothing, three times, and proved each layer
before moving on.

llm-from-scratch v1.0 is done: Learn, Build, and a compressed Deep,
all open, all measured, all on CPU.

Learn: the whole model in pure NumPy, backpropagation written by
hand. Loss fell from 3.2137 to 0.1330 over 300 steps, but the loss
was never the proof. The proof is the gradient check: analytic
gradients against finite differences, agreement below 1e-4 on every
parameter tensor. 104,704 parameters, 8 tests, and a second run that
reproduced the output exactly.

Build: the same model in PyTorch and in JAX, 110,916 parameters
each, same 20,000-character Hamlet slice, same recipe, checkpoint
restart at step 150 of 300 on both tracks. Validation loss:
4.3464 to 2.0351 in PyTorch, 4.2430 to 2.4048 in JAX. Training
speed on CPU: 7,716.7 tokens per second PyTorch, 11,962.7 JAX.

Deep (compressed): a KV cache study on seeded untrained weights.
Greedy 64 tokens, mean of 5 runs. Config A: 284.4 tokens per second
naive, 664.0 cached. Config B: 78.6 naive, 290.4 cached. Cached
output token-identical to naive over 32 tokens on both configs.
And the awkward result, stated plainly: hand-written attention beat
SDPA in prefill in the recorded run, and the winner flipped between
runs, so I claim no stable prefill winner. 24 tests in total.

The honest limits, in the post and not in a footnote: tiny models,
tiny corpora, CPU only, no GPU or TPU number claimed, and the JAX
track is core JAX only, with Flax, Optax, and Orbax named as v1.1
backlog, not done. The rest of the Deep scope is in the repo
ROADMAP.md.

If you build with LLM frameworks: when did you last check what the
framework is doing under your model?

## First comment (post this as the first comment, per the standing rule)

Repo, releases v0.1, v0.5, and v1.0, and the full tutorial: https://github.com/JesunAhmadUshno/llm-from-scratch
