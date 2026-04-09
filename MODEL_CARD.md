---
language:
- en
license: mit
library_name: transformers
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
- animation
- lottie
- svg
- animtoon
- vector-animation
- text-to-animation
datasets:
- OmniLottie/MMLottie-2M
pipeline_tag: text-generation
---

# AnimTOON-3B: Token-Efficient Vector Animation Generation

**5-7x fewer tokens than OmniLottie (CVPR 2026) for generating Lottie animations.**

| | AnimTOON | OmniLottie |
|---|---|---|
| **Tokens (simple)** | **166** | 616 |
| **Tokens (complex)** | **597** | 4095 |
| **VRAM** | **5GB** | 15.2GB |
| **FPS** | **30** | 8 |
| **Model Size** | **3B LoRA** | 4B full |
| **Custom Tokenizer** | **No** | Yes (40k tokens) |
| **Accepts SVG** | **Yes** | No |

## What is AnimTOON?

AnimTOON is a compact, plain-text animation format that any LLM can generate. Instead of outputting 18,000+ tokens of raw Lottie JSON, AnimTOON describes animations in ~166-597 tokens of human-readable text.

```
anim fr=30 dur=120

layer Logo shape
  fill #000000
  path sh x2
  pos [0.5,0.5]
  rot 0.0->-67 0.04->46 0.14->-31 0.28->0 ease=bounce
  scale 0.0->[0,0] 0.14->[90,90] 0.28->[100,100] ease=smooth
  opacity 0.0->0 0.14->100 ease=fade
```

This produces a complete animated .lottie file with bounce entrance, rotation wobble, and fade-in.

## How to Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("srk0102200/AnimTOON-3B")
model = AutoModelForCausalLM.from_pretrained(
    "srk0102200/AnimTOON-3B",
    dtype=torch.float16,
    device_map="cuda"
)

prompt = "a red circle pulsing in the center with a smooth bounce"
messages = [{"role": "user", "content": f"Generate AnimTOON animation: {prompt}"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda")

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
result = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(result)
```

## Convert to .lottie

```python
# Clone: git clone https://github.com/srk0102/svg-animator.git
import sys; sys.path.insert(0, 'src')
from toon_animator import animtoon_to_dotlottie_full

animtoon_to_dotlottie_full(result, "output.lottie")
# Preview at https://lottiefiles.com/preview
```

## Animate Any SVG

```python
from lottie import parsers  # pip install lottie

# Convert SVG to Lottie (perfect paths)
anim = parsers.svg.parse_svg_file("your_logo.svg")
lottie_dict = anim.to_dict()

# Generate AnimTOON animations with the model
# Apply animations to the Lottie layers
# Output: .lottie file with real SVG shapes + AI animations
```

See full pipeline: [test_svg_pipeline.py](https://github.com/srk0102/svg-animator/blob/master/test_svg_pipeline.py)

## Benchmark Results (Measured)

**Same prompt, same hardware:**

| Test | AnimTOON Tokens | OmniLottie Tokens | Ratio |
|------|----------------|-------------------|-------|
| Apple logo bounce | 207 (41 shape + 166 anim) | 1113 | 5.4x fewer |
| Smiley face complex | 597 | 4095 | 6.9x fewer |
| Simple ball bounce | 176 | 616 | 3.5x fewer |

**Dataset statistics (99,650 samples):**
- Average raw Lottie JSON: 18,202 tokens
- Average AnimTOON: 222 tokens
- Token reduction: 98.8%

## Current Status (v3)

**v3 adds character animation support** trained on Spine + DragonBones skeletal data.

The model now works for:
- Icon/logo animations (pulse, bounce, spin, fade, wobble)
- **Character idle/walk cycles (14 layers, coordinated)**
- **Multi-part SVG animation (47-part crab demo)**
- Correct color matching from text descriptions
- SVG + animation pipeline with per-part anchor points

**Limitations:**
- No shape generation (requires SVG input)
- Model output varies between runs (temperature-dependent)
- Position animation on shape groups not yet supported
- Not yet trained on facial expressions

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen/Qwen2.5-3B-Instruct |
| Method | LoRA (r=16, alpha=32) merged into base |
| Training Data | 99,650 (MMLottie-2M) + 10,000 (layer-aware) + 984 (Spine/DragonBones) |
| Hardware | 1x NVIDIA RTX 5060 Ti (16GB) |
| Framework | Unsloth |
| Token Reduction | 98.8% vs raw Lottie JSON |

## Links

- **GitHub:** [github.com/srk0102/svg-animator](https://github.com/srk0102/svg-animator)
- **OmniLottie (comparison):** [arxiv.org/abs/2603.02138](https://arxiv.org/abs/2603.02138)
- **MMLottie-2M Dataset:** [huggingface.co/datasets/OmniLottie/MMLottie-2M](https://huggingface.co/datasets/OmniLottie/MMLottie-2M)

## Citation

```bibtex
@misc{sivaramakrishna2026animtoon,
  title={AnimTOON: Token-Efficient Vector Animation Generation via Compact Text Format},
  author={Siva RamaKrishna},
  year={2026},
  url={https://github.com/srk0102/svg-animator}
}
```

## License

MIT License - see [LICENSE](https://github.com/srk0102/svg-animator/blob/master/LICENSE)
