# llm-from-scratch

**Question.** Can one small GPT be built three ways (NumPy with
hand-written backpropagation, PyTorch, core JAX) so that each is proven
correct by test, and how much does a KV cache speed up greedy generation
of that model on a CPU?

**Headline.** On one CPU thread, KV-cached greedy generation ran 1.45x ± 0.01 (117,060 parameters) and 1.42x ± 0.01 (827,204 parameters) faster than naive generation (mean ± std over 5 trials, spark-6aa3 aarch64, torch 2.12.0+cu130 on CPU), with identical output in every trial; with 20 threads the speedup was 1.88x ± 0.52 and 0.49x ± 0.11. The earlier single-run 2.33x and 3.69x came from a different machine and torch build (Muse VM, torch 2.15.0+cpu) and have not been re-measured under the same conditions.

**Field.** ML and AI research. **Job family.** Machine learning
engineering (LLM internals and inference).

**Verdict.** Functional, not pre-registered. Tests pass in CI on Python
3.11 and 3.12, one command runs every test, and each level's runner has
run end to end with its results file in this repo. No plan with
hypotheses and pass lines was locked before the work, so the verdict is
capped at Functional; the speed question needs a locked confirmatory
sweep (PROPOSAL.md section 4).

Live dashboard: https://jesunahmadushno.github.io/llm-from-scratch/
(train the NumPy model's JavaScript port in your browser, plus every
recorded result, read from this repo at build time).

## Reproduce

```text
pip install numpy
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install jax==0.12.0.1
python -m unittest discover -s tests -p "test_*.py" -v
```

Re-run each level (each runner rewrites its results file beside it):

```text
python level-1-learn/train_tiny.py
python level-2-build/run_build.py
python level-3-deep/run_deep.py
OMP_NUM_THREADS=1 python level-3-deep/run_deep.py --variance 5
```

Read the headline numbers from the evidence:

```text
python3 -c "import json;[print(t,[(c['params'],c['cache_speedup']['mean'],c['cache_speedup']['std'],c['greedy_cached_equals_naive_all']) for c in json.load(open(f'level-3-deep/results_deep_variance_{t}.json'))['configs']]) for t in ('t1','t20')]"
```

CPU only, no GPU or paid service. The tests take about 30 seconds; the
dashboard check (`cd site && npm ci && node src/check.mjs`) fails if a
README table drifts from its results file.

## Lifecycle

- Question: DONE 2026-11-08
- Proposal: OPEN
- Baseline: DONE 2026-11-07
- Build: DONE 2026-11-04
- Evaluate: OPEN
- Report: OPEN
- Review: OPEN
- Release: OPEN

The question was written at migration, after the work: the 2026-11-04
scaffold set deliverables, not a question. No proposal was locked
(PROPOSAL.md says why). The baseline is the n-gram cross-entropy on the
Build split. Evaluate stays open until the field profile is met:
repeated seeds for Build, ablations, and the locked sweep. The tags
v0.1, v0.5 and v1.0 are history from the old level ladder (CHANGELOG.md).

## Layout

| Path | What it holds |
| --- | --- |
| PROPOSAL.md | Question, scope as committed at the scaffold, prior work, an unlocked confirmatory plan (not pre-registered) |
| REPORT.md | Background, method, results, expectations scored with misses, threats, review findings, future work |
| level-1-learn/ | Code: NumPy GPT with hand-written backprop, its runner; evidence: results_learn.json; method write-up: README.md |
| level-2-build/ | Code: PyTorch and core JAX ports, shared data (corpus.txt), runner; evidence: results_build.json; README.md and BUILD-SPEC.md (design notes written with the build) |
| level-3-deep/ | Code: KV cache and attention study, runner; evidence: results_deep.json and results_deep_variance_t1/_t20.json; README.md and DEEP-SPEC.md |
| tests/ | Unit tests for each level and the layout test, run in CI |
| docs/decisions/ | Records of hard-to-reverse choices |
| site/ | The live dashboard, built from the files above; its check fails if README tables drift from the results files |
| fanout/ | Drafts (blog, tutorial, website, LinkedIn, paper brief), release notes, and the publicity log; nothing posted |
| CHANGELOG.md, CITATION.cff, SECURITY.md, LICENSE | Change history, citation, security policy, licence |

There is no src/, experiments/ or results/ folder: code, raw evidence
and results sit together in each level folder, because each runner
writes its results file beside itself and the dashboard reads them
there. Each results file is the raw output of one dated run (date,
machine and versions inside); there is no separate scoring step.

## License

Code: Apache-2.0 (see LICENSE). Tutorials: CC BY 4.0
(docs/decisions/0001).
