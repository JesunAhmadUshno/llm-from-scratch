"""Run both Build framework tracks on the same corpus and print the
side-by-side table. Usage: python level-2-build/run_build.py
Requires: numpy, torch (CPU), jax (CPU). See README for install lines.
"""

import json
import os
import resource
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import CharTokenizer, load_corpus  # noqa: E402


def main():
    text = load_corpus()
    tokenizer = CharTokenizer(text)
    here = os.path.dirname(os.path.abspath(__file__))
    results = {"corpus_chars": len(text), "vocab_size": tokenizer.vocab_size}
    from torch_gpt import train_torch
    torch_result, _ = train_torch(
        text, tokenizer, os.path.join(here, "checkpoint_torch.pt"))
    results["pytorch"] = torch_result
    from jax_gpt import train_jax
    jax_result, _ = train_jax(
        text, tokenizer, os.path.join(here, "checkpoint_jax.pkl"))
    results["jax"] = jax_result
    results["peak_rss_mb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    print(json.dumps(results, indent=2))
    for name in ("pytorch", "jax"):
        row = results[name]
        print(f"{name}: params {row['params']}, final loss "
              f"{row['loss_mean_last_20']}, val {row['val_loss_before']} "
              f"to {row['val_loss_after']}, {row['tokens_per_second']} "
              f"tokens per second, restarted at {row['restarted_at']}")
    return results


if __name__ == "__main__":
    main()
