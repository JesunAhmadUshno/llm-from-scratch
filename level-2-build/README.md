# Level 2, Build: llm-from-scratch

Status: DONE 2026-11-04 (ROADMAP-V3-BATCH loop cycle, Sprint 1 under
the December Compression Protocol). Evidence: the code in this
folder, 7 Build unit tests (15 tests total with Learn), and the
measured side-by-side run recorded below. Deep is not started.
Spec for this level: BUILD-SPEC.md in this folder.

## What was built

The Level 1 model, ported to both frameworks SPEC.md names, trained
on the same real text set, with the same training recipe:

1. PyTorch track (torch_gpt.py): the TinyGPT architecture as an
   nn.Module with autograd and torch.optim Adam. Checkpoints save
   model, optimizer, and step (torch.save), and the driver restarts
   from the checkpoint at step 150 of 300.
2. JAX track (jax_gpt.py): the same architecture as a pure function
   over an explicit parameter pytree, with jax.grad, a jax.jit
   compiled training step, and a hand-written Adam over the pytree.
   Its checkpoint is a pickle of the parameters, the Adam state, and
   the step, and it also restarts at step 150 of 300.
3. Shared data (common.py, corpus.txt): Hamlet from Project
   Gutenberg ebook 1524 (public domain), the first 20,000 characters
   starting at ACT I, committed so the run reproduces offline. One
   character tokenizer (68 characters), one 90/10 train and
   validation split, and one seeded batch sampler serve both tracks,
   so the comparison is frameworks, not data.

## Run it

```text
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install jax
python -m unittest discover -s tests -p "test_build*.py" -v
python level-2-build/run_build.py
```

## Measured results (2026-11-04, Muse VM, Python 3.12.3, torch 2.15.0+cpu, jax 0.12.0.1)

Method: identical config for both tracks (2 layers, 4 heads,
embedding 64, block 32, batch 16, Adam at 5e-3, 300 steps, restart
from checkpoint at step 150). Tokens per second counts training
tokens only (steps x batch x block, divided by wall time); the JAX
timing includes the first step's JIT compilation. Validation loss is
the mean over 10 fixed-seed batches from the held-out 10 percent.

| Measure | PyTorch | JAX |
| --- | --- | --- |
| Parameters | 110,916 | 110,916 |
| Train loss, step 1 | 4.4221 | 4.2365 |
| Train loss, mean of last 20 steps | 2.0665 | 2.4782 |
| Validation loss, before training | 4.3464 | 4.2430 |
| Validation loss, after training | 2.0351 | 2.4048 |
| Training wall time (seconds) | 19.90 | 12.84 |
| Tokens per second | 7,716.7 | 11,962.7 |
| Checkpoint restart | step 150, resumed to 300 | step 150, resumed to 300 |

Process peak memory for the combined run: 1,066.4 MB RSS (both
tracks run sequentially in one process; per-track memory is not
claimed).

Loss curve (training loss at the step):

| Step | PyTorch | JAX |
| --- | --- | --- |
| 50 | 2.5250 | 3.1921 |
| 100 | 2.3982 | 2.8447 |
| 150 | 2.2957 | 2.6106 |
| 200 | 2.2344 | 2.5295 |
| 250 | 2.1348 | 2.4692 |
| 300 | 2.0300 | 2.5021 |

Greedy samples from the prompt "HAMLET." plus a newline (40 new
characters): PyTorch produced "HAMLET." then a newline, then
"HORNAMLELLUS." and a line of repeated "the"; JAX produced "HAMLET."
then short repeated name-like fragments ("HO.", "HARARDO."). Read
honestly: at 300 steps on 18,000 training characters both models
learn the play's shape (speaker names in capitals, short lines) and
neither writes English. The full sample strings sit in the runner
output.

Summary claim: on the same 20,000-character Hamlet slice and the
same 300-step recipe, the PyTorch and JAX ports of the Level 1 model
(110,916 parameters each) both restarted from a checkpoint at step
150 and finished with validation loss 2.0351 (PyTorch) and 2.4048
(JAX), at 7,716.7 and 11,962.7 training tokens per second on this
machine.

## Limits, stated plainly

- The JAX track uses core JAX only: no Flax model definition, no
  Optax optimizer, no Orbax checkpoint. SPEC.md names those
  libraries. The mechanisms (parameter tree, jit step, Adam,
  checkpoint restart) are real and tested; the named library layer
  is the gap, and it is a named deduction below and a Deep and
  v1.1 backlog item.
- 300 steps on 18,000 training characters is a framework comparison
  run, not a trained language model. Final losses differ between
  tracks because the parameter inits and batch order are seeded
  per track, not bit-identical across frameworks.
- One configuration, one corpus, CPU only. The run executed on the
  Muse VM, not the assigned BlackBox GPU node, and no GPU number
  is claimed.
- Checkpoint files are run outputs, written beside the code on
  every run and git-ignored, not committed artifacts.

## Next

Level 3, Deep: the KV cache and the hand-written against fused
attention speed study from SPEC.md, plus the written PyTorch vs
JAX comparison (speed, code size, debugging pain, where XLA fusion
helps) and the TPU note.
