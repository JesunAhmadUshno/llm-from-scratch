# llm-from-scratch

A small GPT built from zero in PyTorch, with the same model rebuilt in JAX. Built in three levels with measured results at each level.

- Learn: a tiny GPT in pure NumPy with hand-written backpropagation (2 layers, 4 heads, 104,704 params). Training loss fell from 3.2137 at step 1 to 0.1330 at step 300.
- Build: the same model in PyTorch and in JAX, 110,916 params on each track. Validation loss fell from 4.3464 to 2.0351 in PyTorch and from 4.2430 to 2.4048 in JAX. Training speed was 7,716.7 tokens per second in PyTorch and 11,962.7 tokens per second in JAX.
- Deep (compressed): a KV cache and attention study. Cached generation ran at 664.0 tokens per second against 284.4 tokens per second without the cache on the smaller config, and 290.4 against 78.6 tokens per second on the larger config.

Grades on record (of 10,000): Learn 8,700, Build 8,700, Deep 8,300, fan-out set 9,300.

Every number above is already published in this repo (README.md level write-ups) and reproduced by the runners in this repo, checked in CI. The full write-ups are in the level folders: level-1-learn, level-2-build, level-3-deep. The tutorial and blog post are in fanout/.

Releases: https://github.com/JesunAhmadUshno/llm-from-scratch/releases
