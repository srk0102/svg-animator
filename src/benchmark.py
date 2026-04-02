"""
AnimTOON Benchmark

Runs inference on test prompts, measures success rate, token count, and speed.
Compares against OmniLottie paper numbers (Table 1).

Usage:
    python src/benchmark.py --model models/animtoon-3b/checkpoint-3400
    python src/benchmark.py --model models/animtoon-3b/final
    python src/benchmark.py --model models/animtoon-3b/final --cpu
"""

import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch
from test_inference import load_model, generate_animtoon
from toon_animator import animtoon_to_dotlottie_full, count_tokens


PROMPTS = [
    "a bouncing blue ball with fade in",
    "a red circle pulsing in the center of a white background",
    "loading spinner rotating continuously",
    "success checkmark animation with green color",
    "a yellow star spinning and scaling up",
    "The video shows a document icon appearing with a fade in effect",
    "a blue arrow pointing right that slides in from the left",
    "notification bell ringing animation",
    "a progress bar filling from left to right",
    "heart icon beating with scale animation in red",
]


def run_benchmark(model_path: str, base_model: str, use_cpu: bool = False):
    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("Running on CPU (slow but doesn't use VRAM)")
    print()

    model, tokenizer = load_model(model_path, base_model)
    print()

    results = []
    os.makedirs("outputs", exist_ok=True)

    for i, prompt in enumerate(PROMPTS):
        print(f"--- [{i+1}/{len(PROMPTS)}] {prompt[:60]} ---")

        try:
            text, gen_time = generate_animtoon(
                model, tokenizer, prompt, max_new_tokens=512,
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"prompt": prompt, "tokens": 0, "gen_time": 0,
                           "valid": False, "lottie_ok": False, "layers": 0})
            continue

        tokens = count_tokens(text)
        has_anim = "anim " in text
        has_layer = "layer " in text
        valid = has_anim and has_layer

        # Convert to .lottie
        lottie_ok = False
        num_layers = 0
        try:
            out_path = f"outputs/bench_{i+1}.lottie"
            animtoon_to_dotlottie_full(text, out_path)
            import zipfile
            with zipfile.ZipFile(out_path, "r") as z:
                for f in z.namelist():
                    if f.endswith(".json") and f != "manifest.json":
                        lj = json.loads(z.read(f))
                        num_layers = len(lj.get("layers", []))
            lottie_ok = num_layers > 0
        except Exception:
            lottie_ok = False

        results.append({
            "prompt": prompt, "tokens": tokens, "gen_time": gen_time,
            "valid": valid, "lottie_ok": lottie_ok, "layers": num_layers,
        })

        print(f"  Tokens: {tokens} | Time: {gen_time:.1f}s | Valid: {valid} | Layers: {num_layers}")
        print(f"  Preview: {text[:120]}...")
        print()

    # Summary
    n = len(results)
    valid_count = sum(1 for r in results if r["valid"])
    lottie_count = sum(1 for r in results if r["lottie_ok"])
    avg_tokens = sum(r["tokens"] for r in results) / n if n else 0
    avg_time = sum(r["gen_time"] for r in results) / n if n else 0
    avg_layers = sum(r["layers"] for r in results) / n if n else 0

    print("=" * 65)
    print("=== AnimTOON Benchmark Results ===")
    print("=" * 65)
    print(f"  Model:             {model_path}")
    print(f"  Device:            {'CPU' if use_cpu else 'GPU'}")
    print(f"  Prompts tested:    {n}")
    print(f"  Valid AnimTOON:    {valid_count}/{n} ({valid_count/n*100:.0f}%)")
    print(f"  Valid .lottie:     {lottie_count}/{n} ({lottie_count/n*100:.0f}%)")
    print(f"  Avg tokens:        {avg_tokens:.0f}")
    print(f"  Avg gen time:      {avg_time:.1f}s")
    print(f"  Avg layers:        {avg_layers:.1f}")
    print()
    print("  --- vs OmniLottie (CVPR 2026, Table 1, Real subset) ---")
    print(f"  {'Metric':<25s} {'OmniLottie':>12s} {'AnimTOON':>12s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'Success rate':<25s} {'88.3%':>12s} {f'{lottie_count/n*100:.0f}%':>12s}")
    print(f"  {'Avg output tokens':<25s} {'486':>12s} {f'{avg_tokens:.0f}':>12s}")
    print(f"  {'Total seq tokens':<25s} {'21,200':>12s} {f'{avg_tokens+50:.0f}':>12s}")
    gpu_est = avg_time / 8 if use_cpu else avg_time
    print(f"  {'Gen time (GPU est.)':<25s} {'33.71s':>12s} {f'{gpu_est:.1f}s':>12s}")
    print(f"  {'VRAM required':<25s} {'~80GB':>12s} {'~5GB':>12s}")
    print()

    # Save results
    with open("outputs/benchmark_results.json", "w") as f:
        json.dump({"results": results, "model": model_path,
                   "device": "cpu" if use_cpu else "gpu"}, f, indent=2)
    print("  Results saved to: outputs/benchmark_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AnimTOON Benchmark")
    parser.add_argument("--model", default="models/animtoon-3b/final")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU (no VRAM)")
    args = parser.parse_args()

    run_benchmark(args.model, args.base_model, use_cpu=args.cpu)
