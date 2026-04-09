"""
Generate layer-aware training data from existing AnimTOON training pairs.

Takes existing training data and creates new pairs where:
  Input: SVG layer descriptions (names, colors, positions, shapes)
  Output: Same AnimTOON animation (unchanged)

This teaches the model to generate animations from layer descriptions
instead of only from video descriptions.
"""
import json
import re
import random
import sys

ARROW = "\u2192"
# Mojibake: UTF-8 bytes of → (E2 86 92) decoded as cp1252 gives â†'
# But json.loads on Windows gives: chr(226) + chr(8224) + chr(8217)
ARROW_MOJIBAKE = chr(226) + chr(8224) + chr(8217)

# Animation style descriptions
ANIM_STYLES = [
    "Animate with a smooth entrance effect, each layer appearing one after another.",
    "Animate with a bouncing entrance, layers popping in sequentially.",
    "Animate with fade-in transitions for each layer.",
    "Animate with scale-up entrance, staggered timing between layers.",
    "Animate smoothly with layers appearing in sequence from first to last.",
    "Each layer should animate in with a professional reveal effect.",
    "Animate with smooth transitions, layers fading and scaling in.",
    "Create a clean entrance animation for each layer.",
    "Animate with a gentle reveal, each part appearing with smooth easing.",
    "Layers should animate in order with bounce and fade effects.",
]

# Shape type guesses based on path count
SHAPE_TYPES = ["simple shape", "detailed shape", "icon element", "graphic element", "vector shape"]

# Color name mapping
def hex_to_name(h):
    if not h or not h.startswith('#') or len(h) != 7:
        return None
    r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
    colors = [
        ((255,255,255),"white"), ((0,0,0),"black"), ((255,0,0),"red"),
        ((0,128,0),"green"), ((0,0,255),"blue"), ((255,255,0),"yellow"),
        ((255,165,0),"orange"), ((128,0,128),"purple"), ((255,192,203),"pink"),
        ((128,128,128),"gray"), ((165,42,42),"brown"), ((0,255,255),"cyan"),
        ((0,128,128),"teal"), ((0,0,128),"navy"), ((255,215,0),"gold"),
    ]
    best = "colored"
    best_d = float('inf')
    for (cr,cg,cb), name in colors:
        d = (r-cr)**2 + (g-cg)**2 + (b-cb)**2
        if d < best_d:
            best_d = d
            best = name
    return best


def pos_to_words(pos_str):
    """Convert position like [0.5,0.3] to words."""
    m = re.search(r'\[([0-9.]+),([0-9.]+)\]', pos_str)
    if not m:
        return "center"
    x, y = float(m.group(1)), float(m.group(2))
    h = "left" if x < 0.33 else "right" if x > 0.66 else "center"
    v = "top" if y < 0.33 else "bottom" if y > 0.66 else "middle"
    if h == "center" and v == "middle":
        return "center"
    if h == "center":
        return v
    if v == "middle":
        return h + " side"
    return f"{v}-{h}"


def parse_animtoon_layers(output_text):
    """Extract layer info from AnimTOON output text."""
    layers = []
    current = None

    for line in output_text.split('\n'):
        line = line.strip()

        if line.startswith('layer '):
            if current:
                layers.append(current)
            parts = line.split()
            name = parts[1] if len(parts) > 1 else "Shape"
            ltype = parts[2] if len(parts) > 2 else "shape"
            current = {
                'name': name,
                'type': ltype,
                'fill': None,
                'stroke': None,
                'pos': 'center',
                'path_count': 1,
                'has_animation': False,
            }

        elif current:
            if line.startswith('fill #'):
                current['fill'] = line.split()[1]
            elif line.startswith('stroke #'):
                current['stroke'] = line.split()[1]
            elif line.startswith('pos '):
                pos_val = line[4:].strip()
                if pos_val.startswith('['):
                    current['pos'] = pos_to_words(pos_val)
                elif ARROW in pos_val or ARROW_MOJIBAKE in pos_val:
                    current['has_animation'] = True
                    current['pos'] = "center"
            elif line.startswith('path '):
                m = re.search(r'x(\d+)', line)
                if m:
                    current['path_count'] = int(m.group(1))
            elif ARROW in line or ARROW_MOJIBAKE in line:
                current['has_animation'] = True

    if current:
        layers.append(current)

    return layers


def build_layer_description(layers):
    """Build SVG layer description from parsed layers."""
    n = len(layers)
    if n == 0:
        return None

    desc = f"SVG has {n} layers:\n"

    for i, layer in enumerate(layers):
        color = hex_to_name(layer['fill']) if layer['fill'] else \
                hex_to_name(layer['stroke']) if layer['stroke'] else "colored"
        shape = random.choice(SHAPE_TYPES)
        pos = layer['pos']
        name = layer['name'].replace('Outlines', '').replace('Layer', 'Part ')

        desc += f"  {i+1}. {name} ({color} {shape}, {pos})\n"

    desc += random.choice(ANIM_STYLES)
    return desc


def generate_layer_data(input_file, output_file, max_samples=10000):
    """Generate layer-aware training data from existing pairs."""
    pairs = []
    skipped = 0

    with open(input_file, encoding='utf-8-sig') as f:
        for line in f:
            rec = json.loads(line.strip())
            output = rec['output']

            # Only use samples with actual animations
            if ARROW not in output and ARROW_MOJIBAKE not in output:
                skipped += 1
                continue

            # Parse layers from output
            layers = parse_animtoon_layers(output)

            # Skip samples with 0-1 layers (not useful for layer-aware training)
            if len(layers) < 2:
                skipped += 1
                continue

            # Skip samples with no animated layers
            animated = sum(1 for l in layers if l['has_animation'])
            if animated == 0:
                skipped += 1
                continue

            # Build layer description as new input
            layer_desc = build_layer_description(layers)
            if not layer_desc:
                skipped += 1
                continue

            pairs.append({
                'input': layer_desc,
                'output': output,
            })

            if len(pairs) >= max_samples:
                break

            if len(pairs) % 1000 == 0:
                print(f"  Generated {len(pairs)} pairs (skipped {skipped})...")

    # Shuffle
    random.seed(42)
    random.shuffle(pairs)

    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    print(f"\nDone: {len(pairs)} layer-aware training pairs saved to {output_file}")
    print(f"Skipped: {skipped} (no layers, no animation, or single layer)")

    # Show examples
    print("\n=== Example pairs ===")
    for i in range(min(3, len(pairs))):
        print(f"\n--- Example {i+1} ---")
        print(f"INPUT:\n{pairs[i]['input'][:300]}")
        print(f"\nOUTPUT:\n{pairs[i]['output'][:200]}...")


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/animtoon_train_full.jsonl"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "data/animtoon_layer_aware_10k.jsonl"
    max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 10000

    print(f"Generating layer-aware training data...")
    print(f"  Source: {input_file}")
    print(f"  Output: {output_file}")
    print(f"  Max samples: {max_samples}")
    print()

    generate_layer_data(input_file, output_file, max_samples)
