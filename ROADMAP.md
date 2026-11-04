# ROADMAP: llm-from-scratch v1.1 backlog

v1.0 (2026-11-04) closes Learn, Build, and compressed Deep.
Compressed Deep delivered the single hardest Deep capability (the
KV cache plus the hand-written against fused attention speed
study in PyTorch), plus the written PyTorch vs JAX comparison,
the debugging log, and the TPU note in the Deep README. The items
below are the rest of the v2 Deep scope, named here as the v1.1
backlog. None of them is done, and none is claimed.

## v1.1 backlog

1. Flax, Optax, and Orbax library layer: rebuild the JAX track's
   model definition in Flax, its optimizer in Optax, and its
   checkpointing in Orbax, replacing the hand-written pytree,
   Adam, and pickle checkpoint from Build, and re-run the
   side-by-side Build comparison on that basis.
2. JAX KV cache: the Deep cache study in JAX, with the generation
   loop expressed for jit (fixed maximum length, lax.scan, mask),
   so the PyTorch and JAX cache numbers can be compared directly.
3. Larger and longer training: train past the 300-step comparison
   recipe, on a larger corpus and at larger configs, so the
   framework comparison includes a model that actually learns the
   text, with held-out loss reported per track.
4. GPU benchmarks: repeat the Deep generation and prefill study
   on a GPU node, where fused attention is expected to matter at
   larger batch and sequence sizes, with the same naive, cached,
   hand-written, and SDPA discipline and measured numbers only.
5. Profiler trace for the XLA fusion claim: the Deep README
   states the fusion mechanism as reasoning from timing. A jax
   profiler trace showing the fused kernels would turn that
   reasoning into evidence.
6. TPU run: run the JAX track on a real TPU when a free route
   exists, with static-shape generation and bfloat16, and replace
   the reasoning-only TPU note with measured numbers. No route
   exists in the current environment and no credential is created
   by this project on its own.
7. Cache systems study: paged or evicting cache variants, with a
   windowed naive mode as the new correctness reference, plus
   sampling (temperature and top-k) equivalence under caching,
   which Deep v1.0 studied for greedy only.
8. Workshop paper submission: the PyTorch vs JAX same-model study
   is the paper candidate named in SPEC.md. The numbers are
   reproducible from this repo at v1.0. Submission itself is
   Jesun's tap and is not part of any automated run.

## Standing rules for v1.1 work

- Greedy cached output stays token-identical to the reference
  mode through every change, or the change does not land.
- Every new number in a README comes from a run executed in that
  cycle, with its machine and method stated. Nothing is carried
  forward as if it were freshly measured.
- No GPU or TPU number is published without a run on that
  hardware in the same cycle.
