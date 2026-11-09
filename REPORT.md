# REPORT: llm-from-scratch (P01)

Field profile: ML and AI research. Verdict: Functional, not
pre-registered (see README and PROPOSAL.md). Every number below is read
from a results file in this repo; where a number has no file behind it,
this report says so.

## Background

One small GPT architecture (character level, pre-LayerNorm, tanh-GELU
MLP) built three times: a pure NumPy model with a hand-written backward
pass, a PyTorch port, and a core JAX port. A KV cache study in PyTorch
then measures what caching buys at this scale on a CPU.

From the dossier at ship time (formerly DOSSIER.md, Present, moved
verbatim apart from an internal hiring score):


Current demand evidence: framework work appears in 27 of 56 sampled postings, LLM work in 33 of 56 [confirmed, fleet and scoring pass 2026-11-04; for P23 to P29, the decision pack 2026-11-04]. The Round 3 research (2026-11-04) found the market centre at post-training, evals, environments, and inference depth [confirmed as the research finding; the report is at ~/workspace/research_notes/roadmap-v3-revaluation-research-20261104-0949/report.md]. Tools and versions are pinned at Build time, not in this v1.
Update 2026-11-04 (ship state, v1.0): Learn, Build, and compressed Deep are done. Ship state is one small GPT architecture built three times: a pure NumPy model with hand-written backpropagation proven by a finite-difference gradient check below 1e-4 (104,704 parameters, loss 3.2137 to 0.1330 over 300 steps, second run identical), the same model in PyTorch and core JAX trained on a 20,000-character Hamlet slice with checkpoint restart at step 150 (110,916 parameters each, validation loss 4.3464 to 2.0351 PyTorch and 4.2430 to 2.4048 JAX, 7,716.7 and 11,962.7 training tokens per second on CPU, combined peak RSS 1,066.4 MB), and a PyTorch KV cache study (greedy 64 tokens from a 14-token prompt, mean of 5: Config A 117,060 parameters at 284.4 naive vs 664.0 cached tokens per second, Config B 827,204 parameters at 78.6 vs 290.4, cached output token-identical to naive over 32 tokens on both configs, hand-written vs SDPA prefill with no stable winner). 24 unit tests pass in total. All runs are CPU only on the Muse VM; no GPU or TPU number is claimed. The v1.1 backlog (Flax, Optax, and Orbax layer, JAX KV cache, larger and longer training, GPU benchmarks, profiler trace, real TPU run) is in ROADMAP.md at the repo root.

## Method

Full method, per level, sits in each level README (the "Measured
results" heading names date, machine and versions):

- Learn (level-1-learn/): NumPy TinyGPT, 2 layers, 4 heads, embedding
  64, block 32, 104,704 parameters, 300 Adam steps on a 167-character
  teaching passage repeated 6 times. Correctness: finite-difference
  gradient check on every parameter tensor at 1e-4 relative error, and
  a causality test.
- Build (level-2-build/): the same model in PyTorch and in core JAX
  (explicit pytree, jax.grad, jax.jit, hand-written Adam), 110,916
  parameters each, the first 20,000 characters of Hamlet (Project
  Gutenberg 1524) from ACT I, one 90/10 split, one seeded sampler, 300
  steps with a checkpoint restart at step 150 on both tracks.
- Deep (level-3-deep/): a PyTorch GPT with hand-written attention;
  naive and KV-cached greedy generation on the same seeded, untrained
  weights (64 tokens from a 14-token prompt, mean of 5), and prefill
  (batch 4, sequence 128, mean of 50) hand-written against
  torch.nn.functional.scaled_dot_product_attention. Config A: 2 layers,
  embedding 64, 117,060 parameters. Config B: 4 layers, embedding 128,
  827,204 parameters.
- Variance pass (2026-11-07): `run_deep.py --variance 5` repeats the
  whole Deep benchmark 5 times, at 1 thread and at 20 threads.
- Baselines (2026-11-07): uniform, add-1 unigram and add-1 bigram
  cross-entropy on the Build split, computed by the dashboard from
  corpus.txt (site/src/lib.js; site/src/check.mjs asserts their order).

## Results

### Learn (2026-11-04, Muse VM, Python 3.12.3, NumPy 1.26.4; evidence level-1-learn/results_learn.json)

| Test | Result |
| --- | --- |
| Tiny GPT training, 300 Adam steps (2 layers, 4 heads, 104,704 params) | Loss 3.2137 at step 1 to 0.1330 at step 300 (mean of last 20 steps 0.1304) |
| Gradient check (analytic vs central finite differences) | Agreement below 1e-4 relative error on sampled entries of every parameter tensor (test enforced) |
| Unit tests | 8 tests pass, including tokenizer roundtrip, causality, training loss decrease, and sampling controls |

Evidence note: results_learn.json is the 2026-11-07 rerun on
spark-6aa3 (NumPy 2.4.5); it reproduces all seven README loss rows and
the three samples exactly (site/src/check.mjs asserts the loss rows).
The 0.13 loss is memorisation of the repeated passage; there is no
held-out split at this level.

### Build (2026-11-04, Muse VM, Python 3.12.3, torch 2.15.0+cpu, jax 0.12.0.1; evidence level-2-build/README.md tables and level-2-build/results_build.json)

| Test | Result |
| --- | --- |
| Same model in PyTorch and JAX, 20,000-char Hamlet slice (Gutenberg 1524), 300 steps, checkpoint restart at step 150 | 110,916 params each; validation loss 4.3464 to 2.0351 (PyTorch) and 4.2430 to 2.4048 (JAX) |
| Speed, training tokens per second on CPU | PyTorch 7,716.7; JAX 11,962.7 |
| Unit tests | 7 Build tests pass (15 total with Learn), including exact checkpoint round-trips on both tracks |

Evidence note: results_build.json is the 2026-11-07 rerun on
spark-6aa3, not the 2026-11-04 run. It reproduces the PyTorch losses
exactly; its JAX validation loss is 2.4369 against the recorded 2.4048.
The 2026-11-04 tokens-per-second figures, the JAX 2.4048 and the
combined 1,066.4 MB peak memory exist only in the Build README table:
no raw results file from that run is in the repo. They are single runs
with no interval.

Baselines on the same split, in nats per character, recomputed from corpus.txt by site/src/lib.js at every dashboard build: uniform
4.2195, add-1 unigram 3.3596, add-1 bigram 2.4929. After 300 steps the
PyTorch track (2.0351) is clearly below the bigram; the JAX track
(2.4048) is only 0.088 below it. The model losses are a mean over 10
validation batches; the baselines score the whole validation split.

### Deep (2026-11-04, Muse VM, torch 2.15.0+cpu, CPU only; evidence level-3-deep/results_deep.json)

| Test | Result |
| --- | --- |
| KV cache study, seeded untrained weights, greedy 64 tokens from a 14-token prompt, mean of 5 generations | Cached generation token-identical to naive over 32 checked tokens on both configs; 664.0 vs 284.4 tokens per second (Config A) and 290.4 vs 78.6 (Config B) |
| Prefill, batch 4, sequence 128, mean of 50 passes | Hand-written attention 0.0105s vs SDPA 0.0183s (Config A) and 0.0424s vs 0.1001s (Config B) in the recorded run; the winner flipped between runs at Config B, so no stable prefill winner is claimed |
| Unit tests | 9 Deep tests pass (24 total with Learn and Build), including cache growth by 1 per step, hand vs SDPA closeness inside 1e-5, and cache causality |

Recorded single-run speedups: 2.33x (Config A; the Deep README prose
says 2.34, the recorded numbers give 664.0 / 284.4 = 2.3347) and 3.69x
(Config B).

### Variance pass (2026-11-07, spark-6aa3 aarch64, Python 3.12.3, torch 2.12.0+cu130 on CPU only; evidence level-3-deep/results_deep_variance_t1.json and _t20.json)

Cache speedup (cached / naive tokens per second), mean ± sample std
over 5 trials; greedy cached output equalled naive in every trial.

| Threads | Config A (117,060 params) | Config B (827,204 params) | Hand-written prefill faster |
| --- | --- | --- | --- |
| 1 | 1.45x ± 0.01 | 1.42x ± 0.01 | 5 of 5 trials, both configs |
| 20 | 1.88x ± 0.52 | 0.49x ± 0.11 | 3 of 5 (A), 0 of 5 (B) |

The recorded 2.33x and 3.69x did not replicate. At these tensor sizes
the wall-clock ratio is dominated by per-call and threading overhead,
not by the attention work the cache removes; at 20 threads on Config B
caching was slower than naive.

## Expectations, scored

None of these was stated before its run (not pre-registered), so all
are exploratory. Misses are shown.

| Expectation (where stated) | Outcome |
| --- | --- |
| E1. Greedy cached generation is token-identical to naive (DEEP-SPEC.md) | HELD: every run and every variance trial, both configs; unit-tested |
| E2. The cache win grows with model size (Deep README, after the run) | MISS: true in the single recorded run (2.33x to 3.69x); not replicated. At 1 thread the ratio is flat (1.45x vs 1.42x); at 20 threads it reverses (1.88x vs 0.49x). Two configs cannot test a trend in any case |
| E3. Fused SDPA beats hand-written attention in prefill (the usual expectation, Deep README) | MISS on CPU at these shapes: hand-written was faster in the recorded run and in 5 of 5 single-thread trials; the winner flips at 20 threads |
| E4. Hand-written NumPy backprop is correct (Learn README) | HELD: finite-difference check below 1e-4, unit-tested in CI |
| E5. JAX trains faster than PyTorch on CPU (Build README) | UNTESTED for uncertainty: 11,962.7 vs 7,716.7 tokens per second in one run; no raw file and no repeat |
| E6. Both framework ports learn the text after 300 steps (BUILD-SPEC.md acceptance: validation loss falls) | HELD for "loss falls"; but JAX ends barely below the bigram baseline, and neither port writes English |

## Threats to validity

- Toy scale: 20,000 characters, 300 steps, CPU only. No GPU or TPU
  number exists in this repo.
- Machines differ between the recorded runs (Muse VM) and the reruns
  (spark-6aa3); timing is not comparable across them.
- Per-track seeding: the PyTorch and JAX losses differ by recipe, not
  only by framework.
- JAX timing includes JIT compilation; memory is one combined process,
  not per track.
- Deep uses untrained weights, one prompt, batch 1, greedy only.
- Provenance: the Learn README records NumPy 1.26.4 and the Deep README
  numpy 2.5.4 for runs on the same VM and day.

## Review findings

### Review of 2026-11-07 (formerly docs/research-review.md; criterion grades removed, grades are internal)

Review date: 2026-11-07. Scope: the repo at the head of `site/live-dashboard`
(Learn, Build and Deep levels, tests, results files).
Each finding cites the file behind it. Gaps
fixed in this pass are marked FIXED; every remaining gap names the exact
compute or data it needs.

#### Summary

| Criterion | One-line reason |
| --- | --- |
| Research question | Clear engineering goals per level, no single falsifiable question |
| Hypotheses | Expectations stated in prose ("cache win grows with size"), none pre-stated as testable |
| Method | Each runner documents config, seeds, timing method; one config per level |
| Baselines | Uniform, unigram and bigram baselines now computed on the Build split |
| Statistics and uncertainty | Variance pass added for Deep; Learn and Build still single runs |
| Reproducibility | Seeded runners, results JSON for all levels, CI on 3.11 and 3.12, versions now pinned |
| Threats to validity | Limits sections are candid; framework confound and toy scale disclosed |
| Related work | PROPOSAL.md section 3 (formerly NOVELTY.md) names "nanoGPT-style builds" only; no positioning |
| Limitations | Every level README has a "Limits, stated plainly" section |
| Engineering rigor | 24 tests incl. a finite-difference gradient check, causality and cache-equivalence tests |

#### 1. Research question

SPEC.md defines deliverables (a NumPy GPT, PyTorch and JAX ports, a KV cache
and an attention speed study), not a question. The closest real question is
the Deep one: "how much does a KV cache speed up greedy generation for a tiny
GPT on CPU, and does fused attention beat hand-written attention at these
shapes?" A sharper version would fix the variable under study (model size,
context length, thread count) and the outcome (tokens per second, ratio).

Remaining gap: write the question and its outcome measure into
a locked PROPOSAL.md before the next run (PROPOSAL.md section 4 holds the unlocked draft). Cost: none.

#### 2. Hypotheses

The Deep README states "the cache win is real and it grows with model size"
after the fact, from two configs. Nothing was stated before the run, and two
points cannot test a trend.

Remaining gap: pre-state H1 "cached/naive tokens-per-second ratio increases
monotonically with n_embd at fixed context" and H2 "the ratio increases with
generated length". Testing them needs the sweep in section 5 (CPU only).

#### 3. Method

Strengths: identical recipe across the PyTorch and JAX tracks (common.py:
same split, sampler, config), explicit timing definitions (training tokens
only; JIT included for JAX), greedy equivalence asserted by the runner, and
untimed warm-up before generation benchmarks.

Weaknesses: one configuration per level for training; per-track seeding means
init and batch order differ across frameworks, so the Build loss gap
(2.0351 vs 2.4048) is a recipe difference, as the README itself says; and
Deep uses untrained weights, one prompt, batch 1, greedy only.

#### 4. Baselines

FIXED: the dashboard computes, in the browser from `corpus.txt` with
`common.split_data` semantics, a uniform baseline (ln 68 = 4.2195 nats/char),
an add-1 unigram baseline (3.3596) and an add-1 bigram baseline (2.4929) on
the same 90/10 split. Against these, after 300 steps the PyTorch track
(val 2.0351) is clearly better than a bigram model, while the JAX track
(2.4048) is only 0.088 nats/char better. That is a useful honest anchor the
README lacked.

Caveat: the model losses are the README's mean over 10 fixed-seed validation
batches; the baselines score the whole validation split. Estimators differ
slightly; the dashboard says so.

Remaining gap: score the trained models on the full validation split with the
same estimator as the baselines. Compute: one Build rerun with an added
full-split evaluation, about 40 s CPU.

#### 5. Statistics and uncertainty

Before this pass every number was a single-run mean with no spread.

FIXED (Deep): `run_deep.py --variance K` repeats the whole benchmark K times
and writes `results_deep_variance_t<threads>.json` with mean, std, min, max.
Recorded on 2026-11-07 (aarch64 CPU, torch 2.12.0+cu130 run on CPU only):

- Single thread (`OMP_NUM_THREADS=1`), 5 trials: cache speedup 1.45x ± 0.01
  (Config A) and 1.42x ± 0.01 (Config B); hand-written prefill faster in 5 of 5
  trials for both configs.
- 20 threads (default), 5 trials: Config A 1.88x ± 0.52; Config B 0.49x ± 0.11,
  i.e. caching was slower than naive.

These do not replicate the recorded 2.33x and 3.69x. The finding is that at
these tensor sizes the wall-clock ratio is dominated by per-call and
threading overhead, not by the attention work the cache removes. The
headline 3.69x should be read as one run on one machine, and the dashboard
labels it that way.

Also found: the Deep README prose says Config A was "2.34 times" faster, but
its own recorded numbers give 2.33 (664.0 / 284.4 = 2.3347, and 0.2250 /
0.0964 = 2.334). The dashboard shows 2.33 and notes the discrepancy; the
build-time check asserts the computed value.

Remaining gaps:
- Learn and Build are single-seed runs. Needed: 5 seeds per track for Build
  (5 x 40 s = about 3.5 min CPU) and report mean ± std of val loss; then the
  PyTorch vs JAX loss comparison can use a common seed set or a paired design.
- Per-repeat timing std inside a trial (kv_cache.benchmark_generation reports
  only the mean of 5). Needed: return the per-repeat times; free.
- Repeat the variance pass on the original Muse VM to separate machine from
  run-to-run noise. Needed: access to that VM, about 2 min.

#### 6. Reproducibility

FIXED:
- `train_tiny.py` now writes `level-1-learn/results_learn.json` (all 300
  losses, samples, versions). A 2026-11-07 rerun on a different machine
  (aarch64, NumPy 2.4.5 against the recorded 1.26.4) reproduced all seven
  README loss rows and the three samples exactly.
- `run_build.py` now writes `level-2-build/results_build.json` with full
  per-step loss arrays. The 2026-11-07 rerun reproduced the PyTorch losses
  exactly at every logged step and the val loss (2.0351); the JAX track
  drifted (val 2.4369 against the recorded 2.4048, mean of last 20 steps
  2.4836 against 2.4782), plausibly from XLA CPU kernel differences across
  platforms. Not yet diagnosed.
- CI now installs `torch==2.15.0` (CPU index) and `jax==0.12.0.1`, the
  versions of the recorded runs, instead of unpinned latest.
- The dashboard's build runs `site/src/check.mjs`, which fails if the README
  tables stop matching the results JSON or the parameter formula stops
  reproducing 104,704 / 110,916 / 117,060 / 827,204.

Remaining gaps:
- The JAX drift above. Needed: rerun Build with jax 0.12.0.1 on x86_64 (a CI
  runner, free) and compare; if it matches the README, the drift is
  platform-specific and should be documented as such.
- Provenance inconsistency: Learn README records NumPy 1.26.4, Deep README
  records numpy 2.5.4 for runs on the same VM and day. Needed: the VM's
  environment record; otherwise state both as recorded.

#### 7. Threats to validity

Disclosed in the READMEs: toy scale, per-track seeding, JIT-inclusive timing,
combined-process memory (1,066.4 MB RSS, not per track although SPEC asks
for per-track memory), shared-VM timing noise, untrained Deep weights.

Not previously disclosed, now on the dashboard: the thread-count sensitivity
of every Deep timing (section 5), and that the Learn level has no held-out
split (its 0.13 loss is memorisation of a 6x repeated 167-character passage).

Remaining gap: per-track peak memory. Needed: run each Build track in its own
process (`/usr/bin/time -v` or `resource` per subprocess), about 40 s CPU.

#### 8. Related work

PROPOSAL.md section 3 (formerly NOVELTY.md) is a DRAFT that names "nanoGPT-style teaching builds" and defers
the comparison. No paper is cited anywhere in the README path.

Remaining gap (no compute, reading only): position the repo against at least
Vaswani et al. 2017 (the transformer), Radford et al. 2019 (GPT-2, whose
tanh-GELU and pre-LayerNorm layout this model uses), Karpathy's minGPT and
nanoGPT repositories (the closest teaching builds), Pope et al. 2022
"Efficiently Scaling Transformer Inference" (KV-cache memory and latency
analysis), and Dao et al. 2022 FlashAttention (why fused attention wins only
at larger shapes on GPU). The likely honest novelty is small: a single model
implemented three ways with a hand-derived backward pass and an exact
cache-equivalence test, measured in public.

#### 9. Limitations

Each level README carries a candid limits section, and the Future work section below lists the
v1.1 backlog (Flax/Optax/Orbax layer, JAX KV cache, longer training, GPU
benchmarks, profiler trace, TPU run, cache systems, paper). The self-assessments that
once sat in docs/index.md were never validation; they have left the
public repo (grades are internal).

#### 10. Engineering rigor

24 unit tests: finite-difference gradient check at 1e-4 relative error on
sampled entries of every parameter tensor, causality, checkpoint round-trips
on both tracks, cache growth by exactly one position per step, greedy
cached = naive. CI runs lint plus all levels on Python 3.11 and 3.12.

The dashboard adds a JavaScript port of the NumPy model whose own gradient
check runs at build time (max relative error about 7e-7 at init) and again
after 60 training steps. One subtlety found while porting: the key-projection
bias has an exactly zero gradient (softmax is invariant to a constant added
to a whole score row), so a ratio-based check on those entries measures only
float noise after training. The JS check counts entries where both gradients
are below 1e-7 as agreeing and says so on the page. The NumPy test avoids the
issue only because it runs at initialisation.

#### Remaining gaps needing compute or data, in priority order

1. Deep scaling sweep for H1 and H2: n_embd in {64, 128, 256, 512}, generated
   length in {32, 64, 128, 512} (block size raised to fit), single thread and
   all threads, 5 trials each. CPU, roughly 10 to 20 minutes. Converts "grows
   with model size" from two points into a curve with error bars.
2. GPU replication of the Deep study (naive vs cached, hand vs SDPA, batch in
   {1, 8, 32}, sequence in {128, 512, 2048}): one GPU node, under an hour.
   This is where fused attention is expected to matter.
3. Build with 5 shared seeds per track and full-split validation: about 4 min
   CPU. Allows a paired PyTorch vs JAX loss comparison with a CI.
4. JAX drift diagnosis on x86_64 CI: free.
5. Per-track memory in separate processes: about 40 s CPU.
6. Longer training on a larger public corpus (Future work item 3): hours on CPU or
   minutes on one GPU; needed before any claim about language modelling
   quality.
7. Related-work section and NOVELTY claim check: reading only.

## Future work (v1.1 backlog, formerly ROADMAP.md, moved verbatim)


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

### Standing rules for v1.1 work

- Greedy cached output stays token-identical to the reference
  mode through every change, or the change does not land.
- Every new number in a README comes from a run executed in that
  cycle, with its machine and method stated. Nothing is carried
  forward as if it were freshly measured.
- No GPU or TPU number is published without a run on that
  hardware in the same cycle.

## Conclusion

The three implementations run end to end and are proven by test: the
NumPy backward pass matches finite differences, both framework ports
restart exactly from a checkpoint, and the KV cache reproduces naive
greedy output token for token. The speed claims do not survive
repetition: on one CPU thread the cache buys about 1.4x at both sizes,
and with 20 threads the result is noise-dominated and can reverse. The
repo answers "does it work" and does not yet answer "how much faster"
with a confirmatory design; PROPOSAL.md section 4 sketches that design.
