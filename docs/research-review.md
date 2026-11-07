# Research review: llm-from-scratch

Review date: 2026-11-07. Scope: the repo at the head of `site/live-dashboard`
(Learn, Build and Deep levels, tests, results files). Grades use a 0 to 5 scale
per criterion, where 3 is "acceptable for a workshop note" and 5 is "PhD
thesis chapter standard". Each grade cites the file that justifies it. Gaps
fixed in this pass are marked FIXED; every remaining gap names the exact
compute or data it needs.

## Summary

| Criterion | Grade | One-line reason |
| --- | --- | --- |
| Research question | 2 | Clear engineering goals per level, no single falsifiable question |
| Hypotheses | 1 | Expectations stated in prose ("cache win grows with size"), none pre-stated as testable |
| Method | 3 | Each runner documents config, seeds, timing method; one config per level |
| Baselines | 3 (was 1) | Uniform, unigram and bigram baselines now computed on the Build split |
| Statistics and uncertainty | 2 (was 0) | Variance pass added for Deep; Learn and Build still single runs |
| Reproducibility | 4 (was 3) | Seeded runners, results JSON for all levels, CI on 3.11 and 3.12, versions now pinned |
| Threats to validity | 3 | Limits sections are candid; framework confound and toy scale disclosed |
| Related work | 1 | NOVELTY.md names "nanoGPT-style builds" only; no positioning |
| Limitations | 4 | Every level README has a "Limits, stated plainly" section |
| Engineering rigor | 4 | 24 tests incl. a finite-difference gradient check, causality and cache-equivalence tests |

## 1. Research question (2/5)

SPEC.md defines deliverables (a NumPy GPT, PyTorch and JAX ports, a KV cache
and an attention speed study), not a question. The closest real question is
the Deep one: "how much does a KV cache speed up greedy generation for a tiny
GPT on CPU, and does fused attention beat hand-written attention at these
shapes?" A sharper version would fix the variable under study (model size,
context length, thread count) and the outcome (tokens per second, ratio).

Remaining gap: write the question and its outcome measure into
`level-3-deep/DEEP-SPEC.md` before the next run. Cost: none.

## 2. Hypotheses (1/5)

The Deep README states "the cache win is real and it grows with model size"
after the fact, from two configs. Nothing was stated before the run, and two
points cannot test a trend.

Remaining gap: pre-state H1 "cached/naive tokens-per-second ratio increases
monotonically with n_embd at fixed context" and H2 "the ratio increases with
generated length". Testing them needs the sweep in section 5 (CPU only).

## 3. Method (3/5)

Strengths: identical recipe across the PyTorch and JAX tracks (common.py:
same split, sampler, config), explicit timing definitions (training tokens
only; JIT included for JAX), greedy equivalence asserted by the runner, and
untimed warm-up before generation benchmarks.

Weaknesses: one configuration per level for training; per-track seeding means
init and batch order differ across frameworks, so the Build loss gap
(2.0351 vs 2.4048) is a recipe difference, as the README itself says; and
Deep uses untrained weights, one prompt, batch 1, greedy only.

## 4. Baselines (3/5, was 1/5)

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

## 5. Statistics and uncertainty (2/5, was 0/5)

Before this pass every number was a single-run mean with no spread.

FIXED (Deep): `run_deep.py --variance K` repeats the whole benchmark K times
and writes `results_deep_variance_t<threads>.json` with mean, std, min, max.
Recorded on 2026-11-07 (aarch64 CPU, torch 2.12.0+cu130 run on CPU only):

- Single thread (`OMP_NUM_THREADS=1`), 5 trials: cache speedup 1.45x ± 0.01
  (Config A) and 1.42x ± 0.01 (Config B); hand-written prefill faster in 5/5
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

## 6. Reproducibility (4/5, was 3/5)

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

## 7. Threats to validity (3/5)

Disclosed in the READMEs: toy scale, per-track seeding, JIT-inclusive timing,
combined-process memory (1,066.4 MB RSS, not per track although SPEC asks
for per-track memory), shared-VM timing noise, untrained Deep weights.

Not previously disclosed, now on the dashboard: the thread-count sensitivity
of every Deep timing (section 5), and that the Learn level has no held-out
split (its 0.13 loss is memorisation of a 6x repeated 167-character passage).

Remaining gap: per-track peak memory. Needed: run each Build track in its own
process (`/usr/bin/time -v` or `resource` per subprocess), about 40 s CPU.

## 8. Related work (1/5)

NOVELTY.md is a DRAFT that names "nanoGPT-style teaching builds" and defers
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

## 9. Limitations (4/5)

Each level README carries a candid limits section, and ROADMAP.md lists the
v1.1 backlog (Flax/Optax/Orbax layer, JAX KV cache, longer training, GPU
benchmarks, profiler trace, TPU run, cache systems, paper). Self-grades in
DOSSIER.md and docs/index.md are self-assessed and are not shown on the
dashboard as validation.

## 10. Engineering rigor (4/5)

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

## Remaining gaps needing compute or data, in priority order

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
6. Longer training on a larger public corpus (ROADMAP item 3): hours on CPU or
   minutes on one GPU; needed before any claim about language modelling
   quality.
7. Related-work section and NOVELTY claim check: reading only.
