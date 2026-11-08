# Security policy

Report a vulnerability privately through GitHub's "Report a
vulnerability" button on this repository's Security tab. Please do not
open a public issue for it. You will get a reply within 7 days.

Scope: the Python code in this repository and the static dashboard in
site/. Nothing here runs as a service; the dashboard runs entirely in
the visitor's browser. No credential is stored in this repository.
Training checkpoints are written locally and git-ignored; the JAX
track's checkpoint is a pickle, so never load a checkpoint file you
did not write yourself.
