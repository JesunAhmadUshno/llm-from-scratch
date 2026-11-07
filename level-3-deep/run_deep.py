"""Run the Deep study: greedy equivalence, generation benchmark
(naive against KV cached), and the prefill benchmark (hand-written
against SDPA). Writes results_deep.json beside this file and prints
it. Usage: python level-3-deep/run_deep.py
Variance pass: python level-3-deep/run_deep.py --variance 5 repeats the
whole benchmark 5 times and writes results_deep_variance_t<threads>.json (mean,
std, min, max per mode and per config) without touching
results_deep.json, which stays the 2026-11-04 record.
Requires: numpy, torch (CPU). See the level README for install lines.
All numbers in the output come from this run, on this machine.
"""

import json
import os
import platform
import resource
import statistics
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "level-2-build"))

import torch  # noqa: E402
from common import CharTokenizer, load_corpus  # noqa: E402
from kv_cache import (  # noqa: E402
    CONFIG_A,
    CONFIG_B,
    DeepGPT,
    benchmark_generation,
    benchmark_prefill,
    generate_cached,
    generate_naive,
)

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


def _spread(values):
    return {"mean": round(statistics.mean(values), 4),
            "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            "min": round(min(values), 4), "max": round(max(values), 4),
            "trials": [round(v, 4) for v in values]}


def variance(trials):
    """Repeat the whole benchmark `trials` times and report the spread.

    The single-run means in results_deep.json moved a lot between runs
    (cache speedup 2.10x to 5.36x, prefill winner flipped), so this
    pass records the spread instead of one point estimate.
    """
    tokenizer = CharTokenizer(load_corpus())
    runs = [[run_config(cfg, tokenizer) for cfg in (CONFIG_A, CONFIG_B)]
            for _ in range(trials)]
    out = {"run_date": date.today().isoformat(),
           "machine": f"{platform.node()} ({platform.machine()})",
           "python": platform.python_version(), "torch": torch.__version__,
           "torch_threads": torch.get_num_threads(),
           "trials": trials, "configs": []}
    for ci, cfg in enumerate((CONFIG_A, CONFIG_B)):
        rows = [run[ci] for run in runs]
        gen = {m: [next(g for g in r["generation"] if g["mode"] == m)
                   for r in rows] for m in ("naive", "cached")}
        pre = {p: [next(g for g in r["prefill"] if g["path"] == p)
                   for r in rows] for p in ("hand", "sdpa")}
        out["configs"].append({
            "config": cfg, "params": rows[0]["params"],
            "greedy_cached_equals_naive_all": all(
                r["greedy_cached_equals_naive"] for r in rows),
            "generation_tokens_per_second": {
                m: _spread([g["tokens_per_second"] for g in gen[m]])
                for m in gen},
            "cache_speedup": _spread([
                c["tokens_per_second"] / n["tokens_per_second"]
                for n, c in zip(gen["naive"], gen["cached"])]),
            "prefill_seconds": {
                p: _spread([g["seconds_mean"] for g in pre[p]]) for p in pre},
            "prefill_hand_faster_trials": sum(
                h["seconds_mean"] < s_["seconds_mean"]
                for h, s_ in zip(pre["hand"], pre["sdpa"])),
        })
    # One file per torch thread count: thread oversubscription on a
    # many-core CPU changes these tiny-tensor timings a lot.
    path = os.path.join(
        HERE, f"results_deep_variance_t{torch.get_num_threads()}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--variance":
        variance(int(sys.argv[2]))
    else:
        main()
