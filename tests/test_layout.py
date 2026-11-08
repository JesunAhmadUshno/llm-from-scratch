"""Portfolio standard checks: the required files exist and no public file carries a grade.

No lock test: the project is not pre-registered (PROPOSAL.md), so there is no locked file to pin.
"""

import os
import re
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED = [
    "README.md", "PROPOSAL.md", "REPORT.md", "CHANGELOG.md", "CITATION.cff", "SECURITY.md", "LICENSE",
    "docs/decisions/0001-license-split.md",
    "level-1-learn/gpt_numpy.py", "level-1-learn/train_tiny.py", "level-1-learn/results_learn.json",
    "level-2-build/run_build.py", "level-2-build/results_build.json", "level-2-build/corpus.txt",
    "level-3-deep/run_deep.py", "level-3-deep/results_deep.json",
    "level-3-deep/results_deep_variance_t1.json", "level-3-deep/results_deep_variance_t20.json",
]

# Grade shapes that must never appear in a public file. Grades are internal.
GRADE = re.compile(
    r"of 10,?000\b|\b\d[\d,]*\s*/\s*10,?000\b|\bGrades?\s*(?::|on record)|\brubric grades?\b"
    r"|Total /1000|\(\d(?:\.\d)?/5(?:,|\))",
    re.IGNORECASE,
)
TEXT = (".md", ".jsx", ".js", ".mjs", ".html", ".cff", ".yml", ".txt", ".py")


def public_files():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    files = out.stdout.split() if out.returncode == 0 else []
    if not files:  # not a git checkout: walk the tree instead
        for d, dirs, names in os.walk(ROOT):
            dirs[:] = [x for x in dirs if x not in (".git", "node_modules", "dist")]
            files += [os.path.relpath(os.path.join(d, n), ROOT) for n in names]
    return [f for f in files if f.endswith(TEXT) and f != "tests/test_layout.py"]


class LayoutTests(unittest.TestCase):
    def test_required_files_exist(self):
        missing = [p for p in REQUIRED if not os.path.exists(os.path.join(ROOT, p))]
        self.assertEqual(missing, [], "missing: " + ", ".join(missing))

    def test_readme_opens_with_the_abstract(self):
        with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
            text = fh.read()
        for field in ("**Question.**", "**Headline.**", "**Field.**", "**Job family.**", "**Verdict.**",
                      "## Reproduce", "## Lifecycle", "## Layout"):
            self.assertIn(field, text)
        self.assertIn("not pre-registered", text.split("**Verdict.**")[1].split("\n\n")[0])

    def test_no_public_grade(self):
        hits = []
        for f in public_files():
            with open(os.path.join(ROOT, f), encoding="utf-8", errors="replace") as fh:
                for n, line in enumerate(fh, 1):
                    if GRADE.search(line):
                        hits.append(f"{f}:{n}: {line.strip()[:80]}")
        self.assertEqual(hits, [], "grades are internal:\n" + "\n".join(hits))

    def test_grade_pattern_catches_known_shapes(self):
        for s in ("9,300 of 10,000", "8700/10000", "Grades on record", "Grade: 3", "## 1. Method (3/5)",
                  "| Total /1000 |"):
            self.assertRegex(s, GRADE)
        for s in ("Math.log10(100000)", "5 of 5 trials", "assertGreater(n, 100000)"):
            self.assertNotRegex(s, GRADE)


if __name__ == "__main__":
    unittest.main()
