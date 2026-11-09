# Level 1, Learn: llm-from-scratch

Status: DONE 2026-11-04 (ROADMAP-V3-BATCH loop cycle). Evidence: the code
in this folder, 8 unit tests in tests/test_learn_gpt.py, and the measured
training run recorded below. Build and Deep are not started.

## The idea in one paragraph

A GPT is a small set of operations repeated: turn characters into
vectors, let each position look at the positions before it (never after
it), mix what it saw through a feed-forward block, and predict the next
character. Frameworks hide those operations behind autograd and fused
kernels. This level hides nothing: the whole model, including the
backward pass, is plain NumPy array code in gpt_numpy.py, so every
number can be checked by hand or by finite differences.

## What is in gpt_numpy.py

- CharTokenizer: character level encode and decode, built from the
  corpus, with a roundtrip test.
- TinyGPT: token plus position embeddings, causal multi-head
  self-attention with an explicit future mask, transformer blocks
  (LayerNorm, attention, GELU MLP, residual paths), a final LayerNorm,
  and a language model head.
- Full manual backpropagation through every layer, plus
  cross_entropy and loss_and_grads.
- Adam: a small optimiser over the parameter dict.
- generate: autoregressive sampling with greedy, temperature, and
  top-k controls.

## The proof that backprop is right

A training loop with a wrong backward pass still runs and still prints
a falling loss sometimes, so a loss curve alone proves little. The
test suite carries the load-bearing check: analytic gradients from the
hand-written backward pass are compared against central finite
differences on sampled entries of every parameter tensor, and must
agree to a relative error below 1e-4. A second test proves causality:
changing the last input token cannot change any earlier logit. If
either test fails, this level is not done, whatever the loss says.

## Run it

```text
pip install numpy
python -m unittest discover -s tests -p "test_learn*.py" -v
python level-1-learn/train_tiny.py
```

Tests need NumPy only. The training run is seeded end to end and
reproduces the numbers below exactly (verified by a second run on
2026-11-04 with identical output).

## Measured results (2026-11-04, Muse VM, Python 3.12.3, NumPy 1.26.4)

Training config: 2 layers, 4 heads, embedding size 64, block size 32,
vocabulary 20 characters, 104,704 parameters, batch 16, Adam at
learning rate 5e-3, 300 steps, about 61 seconds on this VM. The corpus
is a short Hamlet passage repeated 6 times (about 1,000 characters):
a teaching corpus, small on purpose, so the run finishes in about a
minute and the sample can be inspected.

| Step | Loss |
| --- | --- |
| 1 | 3.2137 |
| 50 | 0.8687 |
| 100 | 0.3403 |
| 150 | 0.1475 |
| 200 | 0.1951 |
| 250 | 0.1122 |
| 300 | 0.1330 |

Mean loss over the last 20 steps: 0.1304.

Samples from the prompt "to be or " (40 new characters):

- Before training, greedy: "to be or nstbwbufqqolbmhtfrwufbdbbbbbbbbbbbbbbbbb"
- After training, greedy: "to be or not to be that is the questi whether tis"
- After training, temperature 0.8 with top-k 10: "to be or not to be that is that iohe questi wheth"

Read the numbers honestly: the model memorised a tiny repeated
passage. That is the Learn point (the machinery works, end to end,
and its gradients are proven), not language ability.

## Limits, stated plainly

- The corpus is a small repeated teaching passage, not the real text
  set the SPEC v1 core names. A real corpus and a held-out loss are
  Build level work.
- The model is NumPy, not PyTorch or JAX. PROPOSAL.md defines the same
  model in PyTorch (v1 core) and rebuilt in JAX with Flax (added
  track). Those ports, trained on the same text set with side-by-side
  numbers, are Build level work; this level proves the architecture
  and the maths framework-free first, in the open.
- One training configuration, one seed set, no checkpoint artifact.
  The run is a reproducible script, not a saved model.
- The run executed on the Muse VM (CPU), not the assigned BlackBox
  GPU node. Nothing at this size needs a GPU; the fleet node matters
  from Build, where the framework ports train on a real corpus.

## Next

Level 2, Build: port this exact architecture to PyTorch and to JAX
with Flax per PROPOSAL.md, train both on the same public text set, and
publish side-by-side numbers (tokens per second, memory, final loss)
with an Orbax checkpoint restart on the JAX track.
