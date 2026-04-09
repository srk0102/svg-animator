"""
AnimTOON Inference Script

Loads trained LoRA model, generates AnimTOON from text prompt,
converts to dotLottie, saves output.

Usage:
    python inference.py "a bouncing blue ball"
    python inference.py "spinning red star" --output outputs/star.lottie
    python inference.py "a bouncing blue ball" --raw

Requirements:
    pip install torch transformers peft
"""

import argparse
import json
import os
import sys
import time

# Load .env file for HF_TOKEN
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ.setdefault(key.strip(), val.strip())

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from toon_animator import animtoon_to_dotlottie_full, count_tokens


def load_model(
    model_path: str = "models/animtoon-3b/final",
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
):
    """Load the LoRA-adapted model for inference."""
    print(f"Loading tokenizer from {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=True,
        dtype=torch.float16, device_map="auto",
    )

    print(f"Loading LoRA adapter from {model_path}...")
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    device = next(model.parameters()).device
    print(f"Model loaded on {device}")
    return model, tokenizer


def generate_animtoon(
    model, tokenizer, prompt: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> tuple:
    """Generate AnimTOON text. Returns (text, generation_time_seconds)."""
    messages = [
        {"role": "user", "content": f"Generate AnimTOON animation: {prompt}"},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    print(f"Input tokens: {input_len}")
    print("Generating...")

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_time = time.perf_counter() - t0

    new_tokens = outputs[0][input_len:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    print(f"Generated tokens: {len(new_tokens)}")
    print(f"Generation time:  {gen_time:.2f}s ({len(new_tokens)/gen_time:.1f} tok/s)")
    return result, gen_time


def convert_and_save(animtoon_text: str, output_path: str):
    """Convert AnimTOON text to dotLottie."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print("Converting AnimTOON -> dotLottie...")
    animtoon_to_dotlottie_full(animtoon_text, output_path)

    file_size = os.path.getsize(output_path)
    print(f"  Saved: {output_path} ({file_size:,} bytes)")

    # Save raw AnimTOON text
    txt_path = output_path.replace(".lottie", ".animtoon.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(animtoon_text)
    print(f"  AnimTOON text: {txt_path}")

    # Extract and save Lottie JSON
    try:
        import zipfile
        with zipfile.ZipFile(output_path, "r") as z:
            anim_files = [f for f in z.namelist()
                          if f.endswith(".json") and f != "manifest.json"]
            if anim_files:
                lottie_json = json.loads(z.read(anim_files[0]).decode("utf-8"))
                json_path = output_path.replace(".lottie", ".json")
                with open(json_path, "w") as f:
                    json.dump(lottie_json, f)
                num_layers = len(lottie_json.get("layers", []))
                print(f"  Lottie JSON: {json_path} ({num_layers} layers)")
    except Exception as e:
        print(f"  Could not extract JSON: {e}")


def main():
    parser = argparse.ArgumentParser(description="AnimTOON Inference")
    parser.add_argument("prompt", type=str, help="Animation description")
    parser.add_argument("--model", default="models/animtoon-3b/final")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output", default="outputs/output.lottie")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()

    print("=" * 50)
    print("=== AnimTOON Inference ===")
    print(f"  Prompt: {args.prompt}")
    print(f"  Model:  {args.model}")
    print(f"  Output: {args.output}")
    print("=" * 50)
    print()

    model, tokenizer = load_model(args.model, args.base_model)
    print()

    animtoon_text, gen_time = generate_animtoon(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature, top_p=args.top_p,
    )

    token_count = count_tokens(animtoon_text)
    print()
    print("--- Generated AnimTOON ---")
    print(animtoon_text)
    print("--------------------------")
    print(f"Token count:     {token_count}")
    print(f"Generation time: {gen_time:.2f}s")
    print()

    if args.raw:
        txt_path = args.output.replace(".lottie", ".animtoon.txt")
        os.makedirs(os.path.dirname(txt_path) or ".", exist_ok=True)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(animtoon_text)
        print(f"Saved raw AnimTOON: {txt_path}")
        return

    try:
        convert_and_save(animtoon_text, args.output)
        print()
        print("Done! Open the .lottie file in LottieFiles or any Lottie player.")
    except Exception as e:
        print(f"\nConversion failed: {e}")
        txt_path = args.output.replace(".lottie", ".animtoon.txt")
        os.makedirs(os.path.dirname(txt_path) or ".", exist_ok=True)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(animtoon_text)
        print(f"Raw AnimTOON saved to: {txt_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
