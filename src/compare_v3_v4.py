"""Compare v3-merged vs v4-merged on a battery of prompts.

Writes raw outputs to data/compare_v3_v4/{version}_{i}.animtoon.txt so we
can inspect each one without unicode print issues on Windows console.

Produces a summary of:
  - Does output have 'anim ' header?
  - Layer count
  - Token count
  - Uses parent= (v4 feature)?
  - Uses int pos (v4 feature) or float pos (v3 feature)?
  - Generation time
"""
import argparse
import os
import sys
import time
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from toon_animator import count_tokens

PROMPTS = [
    # Simple
    "a red circle pulsing in the center",
    "a yellow star spinning and scaling",
    # Multi-layer icon
    "a notification bell ringing with the clapper moving left and right",
    "a progress bar with a moving fill from left to right",
    # Basic character
    "a smiling face with two eyes blinking and a mouth",
    "a simple robot with a head that bobs up and down",
    # Rigged character
    "a businessman character waving hello with his right arm",
    "a person walking forward, arms swinging, legs stepping",
    # Complex character
    "a crab with eyes, two claws, and six legs. eyes blink, claws snap, legs move",
    "a cartoon cat with head, body, four legs, and a tail. tail wags, head turns, ears twitch",
]

OUT_DIR = Path("data/compare_v3_v4")


def load_merged(path: str):
    """Load a fully merged model (no LoRA adapter separation)."""
    print(f"  Loading tokenizer: {path}")
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"  Loading model weights: {path}")
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return model, tok


def run_prompt(model, tok, prompt: str, max_new_tokens: int = 1024):
    messages = [{"role": "user", "content": f"Generate AnimTOON animation: {prompt}"}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.5,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tok.pad_token_id,
        )
    gen_time = time.perf_counter() - t0
    new = out[0][input_len:]
    result = tok.decode(new, skip_special_tokens=True).strip()
    return result, len(new), gen_time


def analyze(text: str):
    """Check what features the output uses."""
    has_anim = "anim " in text
    has_layer = "layer " in text
    n_layers = text.count("\nlayer ") + (1 if text.startswith("layer ") else 0)
    has_parent = "parent=" in text
    has_loop = "loop=" in text
    # Detect float vs int positions
    has_float_pos = bool([1 for l in text.split("\n") if l.strip().startswith("pos ") and "." in l])
    has_int_pos = bool([1 for l in text.split("\n") if l.strip().startswith("pos ") and "." not in l])
    tok_count = count_tokens(text)
    return {
        "has_anim": has_anim,
        "has_layer": has_layer,
        "n_layers": n_layers,
        "has_parent": has_parent,
        "has_loop": has_loop,
        "has_float_pos": has_float_pos,
        "has_int_pos": has_int_pos,
        "token_count": tok_count,
    }


def run_version(version: str, model_path: str, report: list):
    print(f"\n{'='*60}")
    print(f"=== {version.upper()} ({model_path}) ===")
    print(f"{'='*60}")

    model, tok = load_merged(model_path)

    for i, prompt in enumerate(PROMPTS):
        print(f"\n  [{i+1}/{len(PROMPTS)}] {prompt[:70]}")
        try:
            text, n_new, gen_time = run_prompt(model, tok, prompt)
        except Exception as e:
            print(f"    FAILED: {e}")
            report.append({
                "version": version, "idx": i+1, "prompt": prompt,
                "error": str(e), "new_tokens": 0, "gen_time": 0,
            })
            continue

        stats = analyze(text)
        stats.update({
            "version": version, "idx": i+1, "prompt": prompt,
            "new_tokens": n_new, "gen_time": round(gen_time, 2),
        })
        report.append(stats)
        print(f"    tokens={n_new:>4}  time={gen_time:>5.1f}s  layers={stats['n_layers']}  "
              f"parent={stats['has_parent']}  int_pos={stats['has_int_pos']}")

        # Save raw output so we can inspect unicode arrows etc.
        out_path = OUT_DIR / f"{version}_{i+1:02d}.animtoon.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")

    # Unload the model to free VRAM before loading the other one
    del model
    del tok
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3", default="models/animtoon-3b-v3-merged")
    parser.add_argument("--v4", default="models/animtoon-3b-v4-merged")
    parser.add_argument("--only", choices=["v3", "v4"], help="run only one")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = []

    if args.only in (None, "v3"):
        run_version("v3", args.v3, report)
    if args.only in (None, "v4"):
        run_version("v4", args.v4, report)

    # Summary table
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    for version in ("v3", "v4"):
        rows = [r for r in report if r["version"] == version]
        if not rows:
            continue
        valid = sum(1 for r in rows if r.get("has_anim") and r.get("has_layer"))
        avg_layers = sum(r.get("n_layers", 0) for r in rows) / len(rows)
        avg_tokens = sum(r.get("new_tokens", 0) for r in rows) / len(rows)
        avg_time = sum(r.get("gen_time", 0) for r in rows) / len(rows)
        w_parent = sum(1 for r in rows if r.get("has_parent"))
        w_int = sum(1 for r in rows if r.get("has_int_pos"))
        w_float = sum(1 for r in rows if r.get("has_float_pos"))
        print(f"\n  {version}:")
        print(f"    Valid outputs:   {valid}/{len(rows)}")
        print(f"    Avg layers:      {avg_layers:.1f}")
        print(f"    Avg new tokens:  {avg_tokens:.0f}")
        print(f"    Avg gen time:    {avg_time:.1f}s")
        print(f"    Uses parent=:    {w_parent}/{len(rows)}")
        print(f"    Uses int pos:    {w_int}/{len(rows)}")
        print(f"    Uses float pos:  {w_float}/{len(rows)}")

    # Full JSON report
    report_path = OUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Full report: {report_path}")
    print(f"  Raw outputs: {OUT_DIR}/v3_*.animtoon.txt and v4_*.animtoon.txt")


if __name__ == "__main__":
    main()
