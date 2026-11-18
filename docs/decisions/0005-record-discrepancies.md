# 0005. Three figures in the level READMEs do not match the record; the corrected values are recorded here

Date: 2026-11-17. Status: accepted.

Context: the preprint's claim check found three discrepancies
between the level READMEs' prose and the repo's own recorded files.
REPORT.md already notes the first and the third. The READMEs are
not edited; this entry is the correction of record.

Decision: the corrected figures are:

1. KV-cache speedup, Config A. level-3-deep/README.md says the
   cache win was "2.34 times" the tokens per second. The recorded
   values in level-3-deep/results_deep.json are 284.4 tokens per
   second (naive) and 664.0 (cached); 664.0 / 284.4 = 2.3347, which
   rounds to 2.33, not 2.34. Correct value: 2.3347.
   fanout/tutorial.md repeats the 2.34 figure.

2. Code size. level-3-deep/README.md says torch_gpt.py is 166
   lines and jax_gpt.py is 224 lines, by wc -l. Counted on
   2026-11-17 with wc -l: level-2-build/torch_gpt.py is 163 lines
   and level-2-build/jax_gpt.py is 223 lines. Correct values: 163
   and 223.

3. NumPy version. level-1-learn/README.md records its measured run
   (2026-11-04, Muse VM, Python 3.12.3) under NumPy 1.26.4;
   level-3-deep/README.md records its measured run (same VM, same
   day, Python 3.12.3, torch 2.15.0+cpu) under numpy 2.5.4. Both
   lines stand as written, and nothing else in the repo settles
   which version the Deep run used: requirements.txt leaves numpy
   unpinned and results_deep.json defers to the level README. This
   discrepancy is recorded unresolved; neither figure is corrected,
   because the record does not support choosing one.

Consequences: quote 2.3347 (2.33 at two decimals) for the Config A
cache speedup and 163 / 223 for the code sizes. The NumPy
provenance of the 2026-11-04 Deep run stays unverified until an
environment record for that run surfaces.
