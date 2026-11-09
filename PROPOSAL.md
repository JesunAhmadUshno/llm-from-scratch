# PROPOSAL: llm-from-scratch (P01)

Status: NOT PRE-REGISTERED. This file is not a locked proposal and
carries no proposal tag. It gathers the plan as it was written, so a
reader can see what was promised before and during the build.

Why there is no lock: SPEC.md was committed in the first commit
(97606fe, 2026-11-04 10:10 UTC), before any code or result. It set
deliverables and acceptance criteria, but no question, no hypotheses,
no pass lines and no analysis plan, so it does not meet the standard's
bar for a pre-registration (docs/METHOD.md stage 2 in the portfolio
index). The level specs (level-2-build/BUILD-SPEC.md,
level-3-deep/DEEP-SPEC.md) were committed in the same commits as the
results they describe. The repo is therefore labelled "not
pre-registered" and capped at Functional until a confirmatory cycle
runs under a newly locked proposal (section 4 below sketches one; it
is not locked).

## 1. Question (written 2026-11-08, after the work)

Can one small GPT be built three ways (NumPy with hand-written
backpropagation, PyTorch, core JAX) so that each is proven correct by
test, and how much does a KV cache speed up greedy generation of that
model on a CPU?

## 2. Scope as committed at the scaffold (formerly SPEC.md, moved verbatim)


Phase skill: spec-driven-development (house engineering pipeline).
Status: Wave 0 scaffold spec, from Roadmap v2 text and the v3 decision pack. Scope and acceptance criteria below are the v2 section text, kept as the source of truth until the project's wave opens its own spec pass.

Fleet node: BlackBox (GPU). Wave: Wave 1.
Hiring evidence: framework work appears in 27 of 56 sampled postings, LLM work in 33 of 56 [fleet pass 2026-11-04].

### Project section text (Roadmap v2, or the v3 candidate text for P23 to P29)

llm-from-scratch: EXPAND

Subject: LLM internals. This project is the PyTorch proof line (Section 2 ruling).
v1 core [confirmed, v1]: a small GPT in pure PyTorch (tokenizer, attention, training loop), trained on a real text set with sampling controls; Deep adds a KV cache and a hand-written against fused attention speed study.
Added track, the same model in JAX (register items 216 to 224):
- Learn: rebuild the same small GPT in JAX with Flax: model build, parameter handling, Optax training state. Use jit, grad, and vmap in real code, not in toy cells.
- Build: train the JAX model on the same public text set as the PyTorch track, with Orbax checkpointing and one restart mid-run. Publish side-by-side numbers: tokens per second, memory, final loss.
- Deep: a written PyTorch vs JAX comparison on the same model (speed, code size, debugging pain, where XLA fusion helps), plus a debugging log of tracer and shape errors and how each was fixed. Close with a TPU note: when TPUs beat GPUs and what changes in code (register item 225).
Fan-out: Book chapter "A Model From Nothing". Paper angle: the PyTorch vs JAX same-model study, a citable comparison note. Course module: P01-llm-from-scratch.

### Acceptance criteria (scaffold level)

- Learn closes when the level-1-learn code runs, its README teaches the idea, and a fresh-eyes reader can follow it.
- Build closes when the system runs end to end, tests pass, and one benchmark or result table with method is in the README.
- Deep closes when the Deep scope in the section text is done, the failure modes are written, and the dossier is updated to ship state.
- No level closes on a claim. Evidence only: code, tests, numbers, links.

## 3. Prior work and proposed novelty (formerly NOVELTY.md and DOSSIER.md Past and Future, moved verbatim)

### NOVELTY.md


Status: DRAFT, PROPOSED. Not verified. This text does not move into any public post, paper, or profile until the Jev claim_check gate passes on the exact text.

#### Proposed claim

Against nanoGPT-style teaching builds and framework tutorials: this project aims to differ in the way its SPEC.md section text defines (a from-zero or self-hosted build, measured in public, on a named home fleet, with the failure modes written down). The exact difference is stated at the project's wave, after the prior art is re-read.

#### Prior art table

| Prior art | What it does | What this project does differently | Measured difference |
| --- | --- | --- | --- |
| nanoGPT-style teaching builds and framework tutorials | See DOSSIER.md, Past | PROPOSED, see the claim above | NO MEASUREMENT YET |

#### Honest size

If the novelty proves small, this file will say so and name the real value instead (a cleaner reproduction, a better benchmark, a first public dataset). No novelty is invented to fill this section.

### DOSSIER.md, Past and Future (the Present section moved to REPORT.md, Background)

#### Past

Prior art for this project is named in NOVELTY.md (nanoGPT-style teaching builds and framework tutorials). The history of the problem, the key papers and systems, and what was tried before are recorded in the Roadmap v2 section text in SPEC.md and will be expanded with dated sources at the project's wave, before Build opens.

#### Future

Open problems and the project's next steps sit in the Deep level scope in SPEC.md. Any forecast about this field is [unverified] unless a dated primary source is added here at the wave update.

Related work named by the 2026-11-07 research review (reading only, not
yet written up): Vaswani et al. 2017 (the transformer); Radford et al.
2019 (GPT-2, whose tanh-GELU and pre-LayerNorm layout this model uses);
Karpathy's minGPT and nanoGPT repositories (the closest teaching
builds); Pope et al. 2022, "Efficiently Scaling Transformer Inference"
(KV-cache memory and latency); Dao et al. 2022, FlashAttention (why
fused attention wins only at larger shapes on GPU). The likely honest
novelty is small: a single model implemented three ways with a
hand-derived backward pass and an exact cache-equivalence test,
measured in public.

## 4. Candidate confirmatory plan (NOT LOCKED)

From the 2026-11-07 research review. If the Owner opens a confirmatory
cycle, this section becomes the seed of a locked PROPOSAL, committed and
tagged before any run.

- H1: the cached / naive tokens-per-second ratio increases
  monotonically with n_embd at fixed context.
- H2: the ratio increases with generated length.
- Design: n_embd in {64, 128, 256, 512}, generated length in {32, 64,
  128, 512} (block size raised to fit), single thread and all threads,
  5 trials each, CPU, about 10 to 20 minutes.
- Build-track comparison: 5 shared seeds per track and full-split
  validation (about 4 minutes CPU), for a paired PyTorch vs JAX loss
  comparison with an interval.
- Pass lines and the CONFIRMED / REFUTED / INCONCLUSIVE rule are to be
  fixed at lock time, not here.
