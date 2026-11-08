# 0002. The JAX track uses core JAX, not Flax, Optax and Orbax

Date: 2026-11-04. Status: accepted (from level-2-build/BUILD-SPEC.md).

Context: the scope names Flax for the model, Optax for the optimiser
and Orbax for checkpoints. None was available in the Build environment.

Decision: write the JAX track as a pure function over an explicit
parameter pytree, with jax.grad, a jax.jit training step, a hand-written
Adam and a pickle checkpoint.

Consequences: the mechanisms are real and tested (exact checkpoint
round-trip), but the named library layer is a gap, carried as item 1 of
the Future work list in REPORT.md. The JAX code is longer than the
PyTorch code partly for this reason, which the code-size comparison in
level-3-deep/README.md must be read against. Pickle checkpoints must not
be loaded from untrusted sources (SECURITY.md).
