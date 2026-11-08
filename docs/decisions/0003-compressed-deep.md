# 0003. Deep was compressed to the KV cache study

Date: 2026-11-04. Status: accepted (from level-3-deep/DEEP-SPEC.md,
December Compression Protocol).

Context: the full Deep scope (JAX KV cache, Flax layer, longer training,
GPU and TPU runs, profiler traces) did not fit the sprint.

Decision: build the single hardest capability in full, the PyTorch KV
cache with the hand-written against fused attention study, and ship the
PyTorch vs JAX comparison, debugging log and TPU note as written
sections. The rest is the Future work list in REPORT.md.

Consequences: no GPU, TPU or JAX-cache number exists, and none is
claimed. The cache has no eviction, so generation past block_size
raises instead of silently diverging from the naive reference.
