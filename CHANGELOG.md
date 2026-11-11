# Changelog

All notable changes to this project. Format: Keep a Changelog 1.1.
Versions follow Semantic Versioning. The tags v0.1, v0.5 and v1.0 were
cut under the old Learn / Build / Deep ladder; under the portfolio
standard v2 the project's verdict is Functional, not pre-registered, so
v1.0 here is a historical tag, not a "v1.0 Released" verdict. Full
release notes for those tags: fanout/release-notes.md.

## [Unreleased]

### Changed
- 2026-11-08: Independent review. The README headline, REPORT variance
  section and dashboard no longer say the earlier 2.33x and 3.69x "did not
  replicate": the five-trial rerun used a different machine and torch build
  (spark-6aa3, torch 2.12.0+cu130 against the Muse VM's 2.15.0+cpu), so it
  is not a same-machine repeat. The claim is lowered to "not re-measured
  under the same conditions".
- 2026-11-08: Adopted the portfolio standard v2. README opens with the
  question, headline, field, job family and verdict (Functional, not
  pre-registered). SPEC.md, NOVELTY.md and DOSSIER.md (Past, Future)
  moved into PROPOSAL.md, which is explicitly not locked. DOSSIER.md
  (Present), the level benchmark tables, docs/research-review.md and
  ROADMAP.md moved into REPORT.md, which scores every expectation and
  shows the misses. PUBLICITY.md moved to fanout/. Public grades
  (docs/index.md, PUBLICITY.md, the research review's criterion
  grades) left the repo: grades are internal. docs/index.md removed;
  its content is in README.md and REPORT.md.
- 2026-11-08: CI on Python 3.11 fixed: jax 0.12.0.1 needs Python 3.12,
  so 3.11 installs jax 0.10.2, the newest release for 3.11
  (docs/decisions/0004). CI gains read-only token permissions, a
  layout test (required files, no public grade) and the dashboard's
  known-answer check on every pull request.
- 2026-11-08: CodeQL also scans the dashboard's JavaScript; Dependabot
  also watches site/ npm packages.

### Added
- 2026-11-08: CHANGELOG.md, CITATION.cff, SECURITY.md,
  docs/decisions/0001 to 0004, tests/test_layout.py.
- 2026-11-07: Live dashboard (site/), results JSON written by the Learn
  and Build runners, Deep variance pass (`run_deep.py --variance K`),
  CI pinned to the recorded torch and jax versions, research review.
- 2026-11-05: Actions polish pack: ruff, CodeQL, Dependabot, release,
  Labeler, Greetings, Stale, Pages docs page.

## [1.0] - 2026-11-04

### Added
- Compressed Deep: PyTorch GPT with hand-written attention, naive and
  KV-cached greedy generation (token-identical), prefill study against
  SDPA, written PyTorch vs JAX comparison, debugging log, TPU note.
  9 Deep tests (24 total).

## [0.5] - 2026-11-04

### Added
- Build: the Learn model in PyTorch and core JAX on a 20,000-character
  Hamlet slice, checkpoint restart at step 150 on both tracks. 7 Build
  tests (15 total).

## [0.1] - 2026-11-04

### Added
- Learn: tiny GPT in pure NumPy with hand-written backpropagation and
  a finite-difference gradient check. 8 tests.
- Scaffold (97606fe): SPEC.md, DOSSIER.md, NOVELTY.md, level folders.

[Unreleased]: https://github.com/JesunAhmadUshno/llm-from-scratch/compare/v1.0...HEAD
[1.0]: https://github.com/JesunAhmadUshno/llm-from-scratch/compare/v0.5...v1.0
[0.5]: https://github.com/JesunAhmadUshno/llm-from-scratch/compare/v0.1...v0.5
[0.1]: https://github.com/JesunAhmadUshno/llm-from-scratch/releases/tag/v0.1
