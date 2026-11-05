# DOSSIER: llm-from-scratch (P01) v1, 2026-11-04

Written under the documentation and publishing standard (past, present, future). Updated at ship time.

## Past

Prior art for this project is named in NOVELTY.md (nanoGPT-style teaching builds and framework tutorials). The history of the problem, the key papers and systems, and what was tried before are recorded in the Roadmap v2 section text in SPEC.md and will be expanded with dated sources at the project's wave, before Build opens.

## Present

Current demand evidence: hiring value 5 of 5 (framework in 27/56, LLM work in 33/56) [confirmed, fleet and scoring pass 2026-11-04; for P23 to P29, the decision pack 2026-11-04]. The Round 3 research (2026-11-04) found the market centre at post-training, evals, environments, and inference depth [confirmed as the research finding; the report is at ~/workspace/research_notes/roadmap-v3-revaluation-research-20261104-0949/report.md]. Tools and versions are pinned at Build time, not in this v1.
Update 2026-11-04 (ship state, v1.0): Learn, Build, and compressed Deep are done. Ship state is one small GPT architecture built three times: a pure NumPy model with hand-written backpropagation proven by a finite-difference gradient check below 1e-4 (104,704 parameters, loss 3.2137 to 0.1330 over 300 steps, second run identical), the same model in PyTorch and core JAX trained on a 20,000-character Hamlet slice with checkpoint restart at step 150 (110,916 parameters each, validation loss 4.3464 to 2.0351 PyTorch and 4.2430 to 2.4048 JAX, 7,716.7 and 11,962.7 training tokens per second on CPU, combined peak RSS 1,066.4 MB), and a PyTorch KV cache study (greedy 64 tokens from a 14-token prompt, mean of 5: Config A 117,060 parameters at 284.4 naive vs 664.0 cached tokens per second, Config B 827,204 parameters at 78.6 vs 290.4, cached output token-identical to naive over 32 tokens on both configs, hand-written vs SDPA prefill with no stable winner). 24 unit tests pass in total. All runs are CPU only on the Muse VM; no GPU or TPU number is claimed. The v1.1 backlog (Flax, Optax, and Orbax layer, JAX KV cache, larger and longer training, GPU benchmarks, profiler trace, real TPU run) is in ROADMAP.md at the repo root.

## Future

Open problems and the project's next steps sit in the Deep level scope in SPEC.md. Any forecast about this field is [unverified] unless a dated primary source is added here at the wave update.
