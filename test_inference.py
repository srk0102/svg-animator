"""
AnimTOON Inference Script

Loads the trained LoRA model, generates AnimTOON from a text prompt,
converts to Lottie JSON, and saves as .lottie file.

Usage:
    python test_inference.py "a bouncing blue ball"
    python test_inference.py "spinning red star with fade in" --output my_anim.lottie
    python test_inference.py "a bouncing blue ball" --model ./models/animtoon-3b
    python test_inference.py "a bouncing blue ball" --max-tokens 2048
    python test_inference.py "a bouncing blue ball" --raw  # print AnimTOON only, no conversion

Requirements:
    pip install torch transformers peft
"""

import argparse
import sys
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from toon_animator import AnimTOONToLottie, lottie_to_dotlottie, count_tokens, validate_and_fix


def load_model(model_path: str, base_model: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Load the LoRA-adapted model for inference."""
    print(f"Loading tokenizer from {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        padding_side="left",  # left-pad for generation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if model_path has a lora_adapter subdirectory
    adapter_path = os.path.join(model_path, "lora_adapter")
    if os.path.exists(adapter_path):
        model_path = adapter_path

    print(f"Loading base model: {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(f"Loading LoRA adapter from {model_path}...")
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    device = next(model.parameters()).device
    print(f"Model loaded on {device}")
    return model, tokenizer


def generate_animtoon(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    """Generate AnimTOON text from a description prompt."""
    messages = [
        {
            "role": "system",
            "content": "You are AnimTOON, a Lottie animation generator. "
                       "Given a description, output the animation in AnimTOON format."
        },
        {
            "role": "user",
            "content": f"Generate AnimTOON animation: {prompt}"
        },
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print(f"Input tokens: {inputs['input_ids'].shape[1]}")
    print("Generating...")

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

    # Decode only the new tokens (skip the input)
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print(f"Generated tokens: {len(new_tokens)}")
    return result


def convert_and_save(animtoon_text: str, output_path: str, fps: int = 60, duration: int = 180):
    """Convert AnimTOON text to Lottie and save as .lottie file."""
    converter = AnimTOONToLottie()

    print("Converting AnimTOON → Lottie JSON...")
    lottie_json = converter.convert(animtoon_text, default_fps=fps, default_duration=duration)

    # Validate and fix common generation issues
    lottie_json, fixes = validate_and_fix(lottie_json)
    if fixes:
        print(f"  Applied {len(fixes)} fix(es):")
        for fix in fixes:
            print(f"    - {fix}")

    num_layers = len(lottie_json.get('layers', []))
    print(f"  Layers: {num_layers}")
    print(f"  Canvas: {lottie_json.get('w', '?')}x{lottie_json.get('h', '?')}")
    print(f"  FPS: {lottie_json.get('fr', '?')}, Frames: {lottie_json.get('op', '?')}")

    # Save as .lottie (dotLottie ZIP format)
    lottie_to_dotlottie(lottie_json, output_path)
    file_size = os.path.getsize(output_path)
    print(f"  Saved: {output_path} ({file_size:,} bytes)")

    # Also save the raw Lottie JSON for inspection
    json_path = output_path.replace('.lottie', '.json')
    import json
    with open(json_path, 'w') as f:
        json.dump(lottie_json, f)
    print(f"  JSON:  {json_path}")

    return lottie_json


def main():
    parser = argparse.ArgumentParser(description="AnimTOON Inference")
    parser.add_argument("prompt", type=str, help="Animation description")
    parser.add_argument("--model", default="./models/animtoon-3b",
                        help="Path to trained model directory")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct",
                        help="Base model name (for tokenizer)")
    parser.add_argument("--output", default="output.lottie",
                        help="Output .lottie file path")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--raw", action="store_true",
                        help="Print raw AnimTOON output without conversion")
    parser.add_argument("--fps", type=int, default=60,
                        help="Animation FPS (default: 60)")
    parser.add_argument("--duration", type=int, default=180,
                        help="Animation duration in frames (default: 180)")
    args = parser.parse_args()

    print("=" * 50)
    print("=== AnimTOON Inference ===")
    print(f"  Prompt: {args.prompt}")
    print(f"  Model:  {args.model}")
    print(f"  Output: {args.output}")
    print("=" * 50)
    print()

    # Load model
    model, tokenizer = load_model(args.model, args.base_model)

    # Generate
    animtoon_text = generate_animtoon(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print()
    print("─── Generated AnimTOON ───")
    print(animtoon_text)
    print("───────────────────────────")
    print(f"AnimTOON tokens: {count_tokens(animtoon_text)}")
    print()

    if args.raw:
        # Save raw AnimTOON text
        txt_path = args.output.replace('.lottie', '.animtoon.txt')
        with open(txt_path, 'w') as f:
            f.write(animtoon_text)
        print(f"Saved raw AnimTOON: {txt_path}")
        return

    # Convert and save
    try:
        convert_and_save(animtoon_text, args.output, fps=args.fps, duration=args.duration)
        print()
        print("Done! Open the .lottie file in LottieFiles or any Lottie player.")
    except Exception as e:
        print(f"\nConversion failed: {e}")
        print("The generated AnimTOON may be malformed. Try:")
        print("  - Lowering temperature (--temperature 0.3)")
        print("  - Increasing max tokens (--max-tokens 4096)")
        print("  - Training for more epochs")

        # Save raw output for debugging
        txt_path = args.output.replace('.lottie', '.animtoon.txt')
        with open(txt_path, 'w') as f:
            f.write(animtoon_text)
        print(f"\nRaw AnimTOON saved to: {txt_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
