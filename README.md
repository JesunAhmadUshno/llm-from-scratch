# llm-from-scratch

A small GPT built from zero in PyTorch, with the same model rebuilt in JAX.

Status: Level 1, Learn DONE 2026-11-04. Level 2, Build DONE 2026-11-04 (ROADMAP-V3-BATCH loop cycle, Sprint 1). Level 3 is scaffold only. Project: P01. Fleet node: BlackBox (GPU). Wave: Wave 1.

## Start here

1. Read SPEC.md for scope and acceptance criteria.
2. Read DOSSIER.md for the past, present, and future of this problem (v1).
3. Read NOVELTY.md for the proposed novelty (DRAFT, pending the Jev claim_check gate).
4. Start at level-1-learn/README.md, then level-2-build, then level-3-deep.

## Level map

- Level 1, Learn: the core idea, built small and clean, with a teaching path.
- Level 2, Build: a real working system other people can run.
- Level 3, Deep: benchmarks, optimization, evaluation, hard cases.

## Benchmarks

Level 1, Learn result (2026-11-04, Muse VM, Python 3.12.3, NumPy 1.26.4; full record in level-1-learn/README.md):

| Test | Result |
| --- | --- |
| Tiny GPT training, 300 Adam steps (2 layers, 4 heads, 104,704 params) | Loss 3.2137 at step 1 to 0.1330 at step 300 (mean of last 20 steps 0.1304) |
| Gradient check (analytic vs central finite differences) | Agreement below 1e-4 relative error on sampled entries of every parameter tensor (test enforced) |
| Unit tests | 8 tests pass, including tokenizer roundtrip, causality, training loss decrease, and sampling controls |

Level 2, Build result (2026-11-04, Muse VM, Python 3.12.3, torch 2.15.0+cpu, jax 0.12.0.1; full record in level-2-build/README.md):

| Test | Result |
| --- | --- |
| Same model in PyTorch and JAX, 20,000-char Hamlet slice (Gutenberg 1524), 300 steps, checkpoint restart at step 150 | 110,916 params each; validation loss 4.3464 to 2.0351 (PyTorch) and 4.2430 to 2.4048 (JAX) |
| Speed, training tokens per second on CPU | PyTorch 7,716.7; JAX 11,962.7 |
| Unit tests | 7 Build tests pass (15 total with Learn), including exact checkpoint round-trips on both tracks |

Deep has no results yet. Its table fills only with measured numbers, with method.

## Novelty

PROPOSED. See NOVELTY.md. The novelty text is a draft until the Jev claim_check gate passes on it.

## License

Code: Apache-2.0 (see LICENSE). Tutorials: CC BY 4.0. Ruled in D-V3-11, 2026-11-04.
