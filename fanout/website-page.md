# Website project page copy: llm-from-scratch

Site: jesunahmadushno.com
Status: COPY ONLY, draft for the project page. Dated 2026-11-04.
Nothing on any website was changed by this file.

## Headline

llm-from-scratch: a GPT built from nothing, proven at every layer

## Summary

llm-from-scratch is an open project by Jesun Ahmad Ushno, P01 in
Roadmap v3. It builds a small GPT three times. First in pure NumPy,
with backpropagation written by hand and proven against finite
differences. Then as the same model in PyTorch and in JAX, trained
side by side on the same Hamlet slice with checkpoint restarts.
Then a KV cache study that measures naive against cached generation
and hand-written attention against fused attention, with the
failures and the flipped result stated in the open.

Status: v1.0, 2026-11-04. Learn, Build, and compressed Deep are
done. Total tests: 24, all passing at v1.0 in CI.

## Results

All results below are measured runs recorded in the level READMEs
in the GitHub repo, Muse VM, CPU only. They prove the machinery,
the comparison method, and the cache mechanism, not language
quality at scale.

| Level | Headline result | Tests |
| --- | --- | --- |
| Learn | NumPy GPT, 104,704 params; loss 3.2137 to 0.1330 over 300 steps (last-20 mean 0.1304); gradient check below 1e-4; second run identical | 8 |
| Build | Same model in PyTorch and JAX, 110,916 params each; Hamlet 20,000-char slice (Gutenberg 1524); restart at step 150 of 300; validation loss 4.3464 to 2.0351 (PyTorch) and 4.2430 to 2.4048 (JAX); 7,716.7 and 11,962.7 training tokens per second on CPU; combined peak RSS 1,066.4 MB | 7 |
| Deep (compressed) | KV cache, greedy 64 tokens from a 14-token prompt, mean of 5; Config A (117,060 params) 284.4 naive vs 664.0 cached tok/s; Config B (827,204 params) 78.6 vs 290.4; cached token-identical to naive over 32 tokens on both configs; hand-written vs SDPA prefill has no stable winner | 9 (24 total) |

## Honest limits

- Tiny models, tiny corpora, CPU only. No GPU or TPU number is
  claimed anywhere in this project.
- The JAX track is core JAX only: no Flax, Optax, or Orbax yet.
  That library layer is named v1.1 backlog.
- Deep used seeded untrained weights; it measures compute, not
  language quality.
- Compressed Deep is not the full v2 Deep scope. The remainder is
  the v1.1 backlog in the repo ROADMAP.md.

## Links

- GitHub repo: https://github.com/JesunAhmadUshno/llm-from-scratch
- Release v0.1 (Learn): https://github.com/JesunAhmadUshno/llm-from-scratch/releases/tag/v0.1
- Release v0.5 (Build): https://github.com/JesunAhmadUshno/llm-from-scratch/releases/tag/v0.5
- Release v1.0 (Learn, Build, compressed Deep): https://github.com/JesunAhmadUshno/llm-from-scratch/releases/tag/v1.0
- Tutorial: fanout/tutorial.md in the GitHub repo
- Blog post draft: fanout/blog-post.md in the GitHub repo

## License note

Code: Apache-2.0. Tutorials: CC BY 4.0.
