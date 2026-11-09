# 0004. CI pins torch and jax to the recorded versions, per Python

Date: 2026-11-07, amended 2026-11-08. Status: accepted.

Context: the 2026-11-04 runs used torch 2.15.0+cpu and jax 0.12.0.1.
Unpinned installs drift. jax 0.12.0.1 requires Python 3.12, so the
Python 3.11 CI job failed at install from 2026-11-07.

Decision: CI installs torch 2.15.0 from the CPU wheel index on both
Pythons, jax 0.12.0.1 on 3.12, and jax 0.10.2 (the newest release that
supports 3.11) on 3.11.

Consequences: the 3.12 job runs the recorded versions; the 3.11 job
proves the code also runs on an older jax, and its runner numbers are
not comparable with the recorded ones. Dependabot cannot see these pins
(they live in ci.yml, not requirements.txt), so they are bumped by hand.
