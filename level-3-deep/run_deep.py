"""Run the Deep study: greedy equivalence, generation benchmark
(naive against KV cached), and the prefill benchmark (hand-written
against SDPA). Writes results_deep.json beside this file and prints
it. Usage: python level-3-deep/run_deep.py
Requires: numpy, torch (CPU). See the level README for install lines.
All numbers in the output come from this run, on this machine.
"""

import json
import os
import resource
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "level-2-build"))

import torch  # noqa: E402

from common import CharTokenizer, load_corpus  # noqa: E402
from kv_cache import (CONFIG_A, CONFIG_B, DeepGPT,  # noqa: E402
                      benchmark_generation, benchmark_prefill,
                      generate_cached, generate_naive)

PROMPT_TEXT = "HAMLET.\nTo be,"
GEN_TOKENS = 64
PREFILL_SEQ = 128
PREFILL_BATCH = 4


def run_config(cfg, tokenizer, seed=1234):
    torch.manual_seed(seed)
    model = DeepGPT(cfg, tokenizer.vocab_size)
    prompt = tokenizer.encode(PROMPT_TEXT)
    naive_ids = generate_naive(model, prompt, 32)
    cached_ids = generate_cached(model, prompt, 32)
    result = {
        "config": cfg,
        "params": model.count_params(),
        "prompt_tokens": len(prompt),
        "equivalence_tokens_checked": 32,
        "greedy_cached_equals_naive": naive_ids == cached_ids,
        "sample_cached": tokenizer.decode(cached_ids),
        "generation": [
            benchmark_generation(model, prompt, GEN_TOKENS, "naive",
                                 repeats=5),
            benchmark_generation(model, prompt, GEN_TOKENS, "cached",
                                 repeats=5),
        ],
        "prefill": [
            benchmark_prefill(model, PREFILL_SEQ, PREFILL_BATCH,
                              use_sdpa=False, repeats=50),
            benchmark_prefill(model, PREFILL_SEQ, PREFILL_BATCH,
                              use_sdpa=True, repeats=50),
        ],
    }
    return result


def main():
    tokenizer = CharTokenizer(load_corpus())
    results = {
        "python_note": "versions recorded in the level README",
        "torch": torch.__version__,
        "vocab_size": tokenizer.vocab_size,
        "prompt_text": PROMPT_TEXT,
        "configs": [run_config(CONFIG_A, tokenizer),
                    run_config(CONFIG_B, tokenizer)],
        "peak_rss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
    for cfg_result in results["configs"]:
        assert cfg_result["greedy_cached_equals_naive"], \
            "cached and naive greedy outputs differ"
    out_path = os.path.join(HERE, "results_deep.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))
    for cfg_result in results["configs"]:
        gen = {row["mode"]: row for row in cfg_result["generation"]}
        pre = {row["path"]: row for row in cfg_result["prefill"]}
        print(
            f"config {cfg_result['config']}: params "
            f"{cfg_result['params']}, equivalence "
            f"{cfg_result['greedy_cached_equals_naive']}, naive "
            f"{gen['naive']['tokens_per_second']} tok/s, cached "
            f"{gen['cached']['tokens_per_second']} tok/s, prefill hand "
            f"{pre['hand']['seconds_mean']}s, sdpa "
            f"{pre['sdpa']['seconds_mean']}s")
    return results


if __name__ == "__main__":
    main()
