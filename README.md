# llm-from-scratch

A small GPT built from zero in PyTorch, with the same model rebuilt in JAX.

Status: Level 1, Learn DONE 2026-11-04 (ROADMAP-V3-BATCH loop cycle). Levels 2 and 3 are scaffold only. Project: P01. Fleet node: BlackBox (GPU). Wave: Wave 1.

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

Build and Deep have no results yet. Their tables fill only with measured numbers, with method.

## Novelty

PROPOSED. See NOVELTY.md. The novelty text is a draft until the Jev claim_check gate passes on it.

## License

Code: Apache-2.0 (see LICENSE). Tutorials: CC BY 4.0. Ruled in D-V3-11, 2026-11-04.
