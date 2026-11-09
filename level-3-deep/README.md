# Level 3, Deep: llm-from-scratch

Status: DONE 2026-11-04 (ROADMAP-V3-BATCH, Sprint 1), as a COMPRESSED
Deep under the December Compression Protocol. Compressed means: the
single hardest capability in the SPEC Deep scope was built, tested,
and benchmarked in full, and the rest of the v2 Deep scope is named
as the v1.1 backlog in REPORT.md (Future work) at the repo root. It is not the
full v2 Deep scope, and this README does not claim it is.
Spec for this level: DEEP-SPEC.md in this folder.

## What was built

1. A PyTorch GPT (kv_cache.py) whose attention is hand-written:
   scores are Q @ K^T / sqrt(head_dim), an explicit causal mask is
   added, softmax normalises, and the output is the weighted sum
   over V. Two generation modes run on the same weights:
   - Naive: every step recomputes the full window.
   - KV cached: a prefill pass over the prompt fills a per layer
     cache of K and V; each later step computes Q, K, V for the one
     new token only, appends its K and V to the cache, and attends
     over the cache. Cache sequence length grows by exactly 1 per
     step (test enforced).
   The context bound fails closed: prompt plus generated tokens
   must fit block_size, and past that the generate functions raise
   ValueError. There is no eviction policy, because a sliding cache
   would silently diverge from the naive mode it is compared with.
2. A fused comparison path on the same weights and shapes:
   torch.nn.functional.scaled_dot_product_attention, benchmarked
   in prefill (batch 4, sequence 128), where the whole sequence is
   processed at once. A test asserts the hand-written output and
   the SDPA output agree on the same random Q, K, V inside a
   stated tolerance (atol 1e-5; the measured maximum difference
   in the development check was 2.38e-07).
3. The written PyTorch vs JAX comparison, the debugging log, and
   the TPU note, as sections of this README below. They draw on
   the Level 2 Build run (the same model in both frameworks) and
   on a JAX probe run in this Deep session.

Weights in the benchmark are seeded and untrained. The study
measures compute, not language quality: the greedy samples in the
runner output are gibberish, as expected from an untrained model,
and they are not quoted here as language output. Greedy
equivalence between the two modes is the correctness claim, and
it is exact: token for token over 32 generated tokens, on both
configs, asserted by the runner and by unit tests.

## Run it

```text
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s tests -p "test_deep*.py" -v
python level-3-deep/run_deep.py
```

The runner writes results_deep.json in this folder. Every number
below comes from the recorded run of that runner in this session
(the runner was executed twice; this is the second run, and the
file on record is that run's output).

## Measured results (2026-11-04, Muse VM, Python 3.12.3, torch 2.15.0+cpu, numpy 2.5.4, CPU only)

Method: greedy generation of 64 tokens from the 14-token prompt
"HAMLET." plus a newline plus "To be,", mean of 5 full
generations per mode. Prefill: batch 4, sequence 128, mean of 50
forward passes per path. Process peak memory for the Deep runner:
274.1 MB RSS.

Config A: 2 layers, 4 heads, embedding 64, block 128, 117,060
parameters.

| Measure | Naive | KV cached |
| --- | --- | --- |
| Generation wall time, mean (seconds) | 0.2250 | 0.0964 |
| Generation tokens per second | 284.4 | 664.0 |

| Prefill path | Wall time, mean (seconds) | Tokens per second |
| --- | --- | --- |
| Hand-written attention | 0.0105 | 48,952.5 |
| SDPA (fused) | 0.0183 | 27,949.6 |

Config B: 4 layers, 4 heads, embedding 128, block 128, 827,204
parameters.

| Measure | Naive | KV cached |
| --- | --- | --- |
| Generation wall time, mean (seconds) | 0.8142 | 0.2204 |
| Generation tokens per second | 78.6 | 290.4 |

| Prefill path | Wall time, mean (seconds) | Tokens per second |
| --- | --- | --- |
| Hand-written attention | 0.0424 | 12,064.0 |
| SDPA (fused) | 0.1001 | 5,115.5 |

Reading the tables honestly:

- The cache win is real and it grows with model size: 2.34 times
  the tokens per second on Config A, 3.69 times on Config B, in
  the recorded run. It is not the textbook quadratic win, because
  at these tiny sizes per-call Python and kernel launch overhead
  is a large share of both modes, and the cached mode still runs
  embeddings, MLP, and the head for every token.
- The prefill result runs against the usual expectation: the
  hand-written path was faster than SDPA at both configs in the
  recorded run. In the first run of the same runner, SDPA was
  faster at Config B (0.0661 against 0.0739 seconds) and slower
  at Config A, so the prefill winner flips between runs at these
  shapes on this shared CPU. No stable prefill winner is claimed.
  The defensible finding is narrower: at batch 4, sequence 128,
  on CPU, fused attention shows no reliable advantage over the
  explicit form, because the tensors are too small for fusion to
  beat its own dispatch overhead. A GPU study is v1.1 backlog.

Summary claim: Level 3 Deep of llm-from-scratch, recorded run
2026-11-04 on the Muse VM CPU (torch 2.15.0+cpu): with seeded
untrained weights, greedy KV-cached generation was token-identical
to greedy naive generation over 32 tokens on both configs, and
generated 64 tokens at 664.0 against 284.4 tokens per second
(Config A, 117,060 parameters) and 290.4 against 78.6 (Config B,
827,204 parameters); prefill at batch 4, sequence 128 took 0.0105
seconds hand-written against 0.0183 SDPA on Config A and 0.0424
against 0.1001 on Config B, a result whose winner flipped between
the two runs of the runner. 9 Deep unit tests pass (24 total with
Learn and Build).

## PyTorch vs JAX comparison (written, from the Level 2 same-model study)

Speed. In the Build run (same 20,000-character Hamlet slice, same
300-step recipe, Muse VM CPU), training ran at 7,716.7 tokens per
second in PyTorch and 11,962.7 in JAX, about 1.55 times faster in
JAX, and the JAX timing includes its first step's JIT compilation
cost, so the steady state gap is at least the printed one. Final
validation losses differed (2.0351 PyTorch, 2.4048 JAX) because
initialisation and batch order are seeded per track, not
bit-identical across frameworks; that is a recipe difference, not
a framework quality result, and neither number should be quoted
as one framework beating the other on learning.

Code size. torch_gpt.py is 166 lines; jax_gpt.py is 224 lines
(wc -l, both including docstrings), with 60 shared lines in
common.py. The JAX track is longer because the parts PyTorch
supplies as library calls are written out: parameter
initialisation as an explicit pytree, a hand-written Adam over
the pytree (about 30 lines), and checkpointing as a hand-rolled
pickle of NumPy arrays. PyTorch gets nn.Module, torch.optim.Adam,
and torch.save in a few lines each. Against that, the JAX forward
function is a single pure function over the pytree, which makes
the model easy to transform (grad, jit, vmap) without touching
the definition, where the PyTorch version carries its behaviour
in module state.

Debugging pain. PyTorch errors are concrete: a shape mismatch
raises at the line that caused it, with the two shapes printed,
in the run you are already watching. JAX errors are one level
more abstract. The load-bearing example is in the debugging log
below: code that works eagerly fails under jit with a
ConcretizationTypeError, because a Python loop bound became a
traced value, and the error names the abstraction, not the line
of model code that created it. The JAX workflow that follows
from this is: debug eagerly with concrete arrays and prints,
then jit the proven function and treat compile errors as a
separate class. PyTorch has no equivalent two-phase split at
this scale, which is its real ergonomic advantage here.

Where XLA fusion helps. The whole JAX training step is one
jit-compiled function, so XLA sees the full graph: the LayerNorm
reductions, the GELU MLP elementwise chain, and the softmax can
be fused into fewer kernels instead of one dispatched operation
per line, and buffers are planned once. That is the plausible
mechanism behind the Build speed gap on CPU at this size, stated
as reasoning from the timing, not as a profiled fact (no profiler
trace was taken; that is a v1.1 item). Where XLA does not help:
data-dependent control flow and dynamic sequence lengths force
either static shapes with padding or a fall back to eager Python
loops, which is exactly the situation in autoregressive
generation, and one reason this Deep cache study was built in
PyTorch, where the dynamic loop is the native style.

## Debugging log (tracer and shape errors, with fixes)

1. JAX tracer error, live in this Deep session (2026-11-04, jax
   0.12.0.1). A probe jitted a generation step whose Python for
   loop ranged over int(n) with n a traced scalar argument. It
   raised ConcretizationTypeError: "Abstract tracer value
   encountered where concrete value is expected: traced array
   with shape int32[]". Fix pattern: the loop length must be a
   static argument (static_argnums) or the loop must be rewritten
   as lax.scan over a fixed maximum length with a mask. This is
   also why the Build JAX generate function is an eager Python
   loop over forward calls, not a jitted function, and it works.
2. SDPA causal mask alignment, caught while writing the fused
   path, before it shipped. F.scaled_dot_product_attention with
   is_causal=True assumes query and key lengths match. In a
   cached step the query length is 1 and the key length is the
   whole cache, so is_causal would build the mask for the wrong
   alignment. Fix in sdpa_attention: is_causal is used only when
   the lengths are equal (prefill); the step case passes
   causal=False, which is correct there because every cached key
   is at or before the query position by construction. The
   step-against-full equivalence test guards this.
3. Hand-written mask shape for the step case, caught while
   writing, before it shipped. The square (seq, seq) mask cannot
   be added to step scores of shape (1, cache length). Fix in
   hand_attention: the mask is applied only when the query length
   is greater than 1, for the same reason as item 2. Guarded by
   the cache growth and equivalence tests.
4. Cache concatenation axis. After the head split the cache
   tensors are laid out (batch, heads, sequence, head_dim), so
   appending a step means torch.cat on dim=2, not the dim=1 that
   the pre-transpose layout (batch, sequence, embedding) would
   suggest. A wrong axis either raises or, worse, silently grows
   the head axis. The cache shape test asserts the sequence axis
   (index 2) grows by exactly 1 per step and the other axes never
   change, so this cannot regress silently.

## TPU note (reasoning only; no TPU was available, no TPU number is claimed)

This section is reasoning from how TPUs and XLA work, not a
measurement. Nothing in it is a benchmark result.

When TPUs are expected to beat GPUs: large dense matrix
multiplication at high batch, in bfloat16, under a fully
XLA-compiled graph with static shapes: large model training and
high-throughput inference serving are the canonical cases. The
TPU matrix unit is built for exactly the QK^T and projection
shapes that dominate a transformer, and a TPU pod adds fast
interconnect for model and data parallel training across many
chips. For a model this small, or for dynamic, Python-driven,
one-token-at-a-time generation on a single sequence, there is no
reason to expect a TPU win: the workload is launch bound, not
matmul bound, the same limit the prefill table above shows on
CPU.

What changes in code, moving the JAX track to a TPU: the model
and training step are already pure functions over a pytree, so
the core code carries over, which is the strongest argument for
the JAX track in this repo. The changes sit around it: devices
are selected and data is sharded explicitly (pjit or jax
sharding APIs, where PyTorch would use a device string and DDP);
shapes must be static or padded to static, so generation needs
a fixed maximum length with masking instead of the Python list
growth both tracks use now; precision is set to bfloat16 by
policy rather than per tensor; and debugging moves further from
the eager style, because the compiled graph runs on the device
as a whole. A real TPU run, with numbers, is v1.1 backlog in
REPORT.md (Future work) and needs a free TPU route that does not exist in
this environment today.

## Failure modes

- Cache memory grows linearly with context, per layer: Config A
  stores 256 fp32 values per token across its layers (about 1 KB
  per token), Config B stores 1,024 (about 4 KB per token). At
  the 128-token block bound that is trivial; scaled to a 100,000
  token context, Config B's cache alone would be about 400 MB
  for one sequence. Long-context caching is a memory system, not
  a free win, and this level's bound hides that by design.
- No eviction: generation past block_size raises instead of
  sliding. A production cache must evict or page, and any
  eviction policy breaks the exact naive equivalence this level
  uses as its correctness proof, so the equivalence would need a
  new proof against a windowed naive mode.
- Near-tie argmax flips: cached and naive compute the same maths
  in different operation orders, so logits can differ in the last
  bits. Greedy outputs were token-identical on every check in
  this level, but a near tie between two tokens could flip on
  that difference and the two modes would then diverge for good.
  Sampling modes were not studied at Deep; greedy only.
- Stale cache: the low-level step API takes the model and the
  cache as separate arguments. Mutating model weights between
  prefill and a step leaves a cache built under the old weights,
  and nothing raises. The generate_cached wrapper cannot hit
  this; direct step callers can.
- No padding or batch masking: a padded batch would attend to
  pad positions in both modes. The study is batch 1 generation
  and fixed batch prefill with no padding, and claims nothing
  beyond that.
- Timing noise: absolute CPU times moved a lot between the two
  runs of the runner on this shared VM (the cache speedup ranged
  from 2.10 to 5.36 times across runs and configs, and the
  prefill winner flipped at Config B). Treat the recorded table
  as one run's measurement with method, not as a constant of
  nature; the ratios and the ordering of the generation modes
  were stable, the absolute numbers were not.

## Limits, stated plainly

- Compressed Deep, by the protocol named above. The rest of the
  v2 Deep scope is REPORT.md (Future work): the Flax, Optax, and Orbax library
  layer, a JAX KV cache, larger and longer training, GPU
  benchmarks, a profiler trace for the XLA fusion claim, and a
  real TPU run.
- Seeded untrained weights, one prompt, greedy only, batch 1
  generation, two configs, CPU only, on the Muse VM, not the
  assigned BlackBox GPU node. No GPU number and no TPU number is
  claimed anywhere in this level.
- The PyTorch vs JAX speed figures are the Level 2 Build training
  numbers, quoted with their date and method, not re-measured at
  Deep. The Deep generation and prefill study is PyTorch only.
