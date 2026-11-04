# SPEC: llm-from-scratch (P01)

Phase skill: spec-driven-development (house engineering pipeline).
Status: Wave 0 scaffold spec, from Roadmap v2 text and the v3 decision pack. Scope and acceptance criteria below are the v2 section text, kept as the source of truth until the project's wave opens its own spec pass.

Fleet node: BlackBox (GPU). Wave: Wave 1.
Hiring evidence: 5 of 5 (framework in 27/56, LLM work in 33/56) [confirmed, fleet and scoring pass 2026-11-04, or the decision pack for P23 to P29].

## Project section text (Roadmap v2, or the v3 candidate text for P23 to P29)

llm-from-scratch: EXPAND

Subject: LLM internals. This project is the PyTorch proof line (Section 2 ruling).
v1 core [confirmed, v1]: a small GPT in pure PyTorch (tokenizer, attention, training loop), trained on a real text set with sampling controls; Deep adds a KV cache and a hand-written against fused attention speed study.
Added track, the same model in JAX (register items 216 to 224):
- Learn: rebuild the same small GPT in JAX with Flax: model build, parameter handling, Optax training state. Use jit, grad, and vmap in real code, not in toy cells.
- Build: train the JAX model on the same public text set as the PyTorch track, with Orbax checkpointing and one restart mid-run. Publish side-by-side numbers: tokens per second, memory, final loss.
- Deep: a written PyTorch vs JAX comparison on the same model (speed, code size, debugging pain, where XLA fusion helps), plus a debugging log of tracer and shape errors and how each was fixed. Close with a TPU note: when TPUs beat GPUs and what changes in code (register item 225).
Fan-out: Book chapter "A Model From Nothing". Paper angle: the PyTorch vs JAX same-model study, a citable comparison note. Course module: P01-llm-from-scratch.

## Acceptance criteria (scaffold level)

- Learn closes when the level-1-learn code runs, its README teaches the idea, and a fresh-eyes reader can follow it.
- Build closes when the system runs end to end, tests pass, and one benchmark or result table with method is in the README.
- Deep closes when the Deep scope in the section text is done, the failure modes are written, and the dossier is updated to ship state.
- No level closes on a claim. Evidence only: code, tests, numbers, links.
