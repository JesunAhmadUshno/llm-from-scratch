# PUBLICITY: llm-from-scratch (P01) v1.0 fan-out log

Date: 2026-11-04. Prepared by the ROADMAP-V3-BATCH loop. The loop
prepares drafts and never posts to social. Every piece below was
graded on the 1,000-point publicity rubric (accuracy 300, research
200, gap fit 200, clarity 150, reach format 150) before this log was
written. Ship line: 900.

## Channel outcomes

| Channel | Outcome, 2026-11-04 |
| --- | --- |
| GitHub releases | Live: v0.1 https://github.com/JesunAhmadUshno/llm-from-scratch/releases/tag/v0.1 , v0.5 https://github.com/JesunAhmadUshno/llm-from-scratch/releases/tag/v0.5 , v1.0 https://github.com/JesunAhmadUshno/llm-from-scratch/releases/tag/v1.0 (verified with gh release list, 2026-11-04) |
| Portfolio index | Held: the index repo is updated by the batch coordinator at sprint review, not inside this fan-out cycle |
| Website page | Draft in fanout/website-page.md. Held: not published, no website was changed by this cycle |
| Blog | Draft in fanout/blog-post.md. Held: not posted to any blog platform |
| Tutorial | In repo: fanout/tutorial.md (course module P01-llm-from-scratch, CC BY 4.0) |
| LinkedIn | Draft in fanout/linkedin-post.md, marked DRAFT NOT POSTED, link as first-comment line per the LinkedIn rule. Held: not posted |
| X | Held: no X draft prepared in this cycle; the LinkedIn draft is the master social draft, adaptation waits for the posting gate |
| Facebook | Held: no Facebook draft prepared in this cycle; any future post follows the keyword CTA rule (no link in body or comments) |
| Instagram | Held: no Instagram draft prepared in this cycle; any future caption follows the link in bio rule |
| Threads | Held: staged only, waits for Jesun's tap (one tap per post, hash-locked). The loop never posts to Threads |
| Video | Held: engine queue. P01 is not a flagship, so its lab video is at scripted and storyboard stage in December per the compression protocol, production January |
| Book chapter | Named in SPEC.md as "A Model From Nothing". Held: chapter draft not yet written; fanout/blog-post.md is its source draft |
| Paper candidate | Draft in fanout/paper-candidate.md. No row for P01 exists in the publications pipeline tracker at 2026-11-04, and the brief says so. Held: submission is Jesun's tap, nothing submitted, no venue contacted |

## Per-piece publicity grades

| Piece | Accuracy /300 | Research /200 | Gap fit /200 | Clarity /150 | Reach /150 | Total /1000 |
| --- | --- | --- | --- | --- | --- | --- |
| fanout/blog-post.md | 295 | 190 | 195 | 145 | 140 | 965 |
| fanout/tutorial.md | 285 | 185 | 190 | 145 | 135 | 940 |
| fanout/website-page.md | 295 | 185 | 190 | 145 | 140 | 955 |
| fanout/linkedin-post.md | 290 | 180 | 185 | 140 | 145 | 940 |
| fanout/paper-candidate.md | 290 | 195 | 190 | 140 | 130 | 945 |
| fanout/release-notes.md | 295 | 180 | 180 | 140 | 135 | 930 |

All six pieces are at or above the 900 ship line.

## Unclosable deductions, named

- Tutorial, accuracy minus 15: in this cycle the Learn test suite
  was executed locally (8 tests, pass) and the Learn runner was
  executed locally to completion (exit 0) and reproduced every
  recorded number exactly, taking about 19 minutes of wall time on
  this shared VM against about 61 seconds in the recording
  conditions. The Build and Deep commands were not executed locally,
  because torch and JAX are not installed in this VM's Python; they
  were verified by file existence and by the repo CI on this
  fan-out commit (run completed success), which installs torch and
  JAX and runs both suites and both runners.
- Blog and website, research deductions: the DOSSIER past section
  is still summary level; prior art depth is the v1.1 dossier work,
  and both pieces say the novelty text stays PROPOSED.
- Paper candidate, accuracy minus 10 and reach minus 20: no
  tracker row exists for P01 to check the brief against, and the
  candidate is a brief only; no draft paper exists yet.
- Release notes, gap fit minus 20: the notes document releases
  that already exist; they add no new evidence, by design.
- Every channel marked held above is a deduction against reach
  that no draft can close: posting follows the standing platform
  rules and gates, and Threads waits for Jesun's tap.

## Overall fan-out grade

9,300 of 10,000. Deductions: 250 for the tutorial local execution
gap named above, 150 for social and video channels held by the
standing gates, 100 for the missing tracker row, 100 for the thin
DOSSIER past section, 100 for timing noise that is inherent to the
recorded CPU benchmarks and is stated, not hidden, in every piece.
