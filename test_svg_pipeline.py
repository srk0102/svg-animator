"""Test full SVG → animated Lottie pipeline."""
import sys
import os
import json
import copy

sys.path.insert(0, 'src')
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Step 1: Use python-lottie for perfect SVG to Lottie
from lottie import parsers

svg_file = sys.argv[1] if len(sys.argv) > 1 else 'inputs/apple.svg'
print(f"SVG: {svg_file}")
anim = parsers.svg.parse_svg_file(svg_file)

# Let python-lottie handle the SVG at its native size
# Then we scale the entire output to 512x512
import xml.etree.ElementTree as ET
tree = ET.parse(svg_file)
root = tree.getroot()
viewbox = root.get('viewBox', '0 0 512 512')
vb = [float(x) for x in viewbox.split()]
vb_w, vb_h = vb[2], vb[3]
scale_factor = 512 / max(vb_w, vb_h) * 100

# Set animation size to match viewBox so paths align
anim.width = vb_w
anim.height = vb_h

lottie_dict = anim.to_dict()

# Now scale canvas to 512x512 and add a wrapper transform
lottie_dict['w'] = 512
lottie_dict['h'] = 512

# Scale each layer's transform to fit 512x512
for layer in lottie_dict.get('layers', []):
    ks = layer.get('ks', {})
    # Set default scale to fit canvas
    ks['s'] = {"a": 0, "k": [scale_factor, scale_factor, 100]}
    # Center the layer
    ks['p'] = {"a": 0, "k": [256, 256, 0]}
    ks['a'] = {"a": 0, "k": [vb_w/2, vb_h/2, 0]}
    layer['ks'] = ks

# Split single layer with multiple shape groups into separate layers
original_layers = lottie_dict.get('layers', [])
layers = []

for orig_layer in original_layers:
    shapes = orig_layer.get('shapes', [])
    if len(shapes) > 1:
        for i, shape in enumerate(shapes):
            new_layer = copy.deepcopy(orig_layer)
            new_layer['shapes'] = [shape]
            new_layer['nm'] = f"Part_{i+1}"
            new_layer['ind'] = len(layers) + 1
            # Scale layer to fit 512x512
            ks = new_layer.get('ks', {})
            ks['s'] = {"a": 0, "k": [scale_factor, scale_factor, 100]}
            new_layer['ks'] = ks
            layers.append(new_layer)
    else:
        orig_layer.get('ks', {})['s'] = {"a": 0, "k": [scale_factor, scale_factor, 100]}
        layers.append(orig_layer)

lottie_dict['layers'] = layers

with open('outputs/robot_static.json', 'w') as f:
    json.dump(lottie_dict, f, indent=2)

print(f"SVG converted: {len(layers)} layers (split from shape groups)")
for i, layer in enumerate(layers):
    print(f"  Layer {i}: {layer.get('nm', 'unnamed')}")

# Step 2: Use a prompt that matches training data format
from prompt_builder import build_prompt_with_svg_info
_, elements = build_prompt_with_svg_info(svg_file, 'entrance')

# Build layer-aware prompt for the new model
prompt = (
    "A cyan-colored elongated shape resembling a smiley face. "
    "The shape rotates gradually around its horizontal axis, "
    "revealing small animated sparkles near the eyes, "
    "enhancing the dynamic visual effect."
)
print(f"\nPrompt: {prompt[:200]}...")

# Step 3: Load model and generate animations
from test_inference import load_model, generate_animtoon

checkpoint = 'models/animtoon-3b-layers/checkpoint-1400'
# Find latest checkpoint
ckpt_dir = 'models/animtoon-3b-layers'
ckpts = [d for d in os.listdir(ckpt_dir) if d.startswith('checkpoint-')]
if ckpts:
    latest = sorted(ckpts, key=lambda x: int(x.split('-')[1]))[-1]
    checkpoint = os.path.join(ckpt_dir, latest)
    print(f"Using latest checkpoint: {checkpoint}")

model, tokenizer = load_model(checkpoint, 'models/animtoon-3b-merged')
text, t = generate_animtoon(model, tokenizer, prompt, max_new_tokens=1024)
print("\nMODEL OUTPUT:")
print(text)

# Step 4: Parse model animations and inject into Lottie JSON
# Handle both real arrow and mojibake versions
arrow = "\u2192"

def has_arrow(line):
    # Check for real arrow, mojibake, or the pattern time→value
    if arrow in line:
        return True
    # Check for any arrow-like pattern: number followed by non-ascii then number/bracket
    import re
    return bool(re.search(r'\d[^\x00-\x7F]+[\d\[\-]', line))

def normalize_arrows(line):
    # Replace any non-ascii sequence between digits that looks like an arrow
    import re
    # Find the actual mojibake bytes in the string
    result = line
    for match in re.finditer(r'(\d(?:\.\d+)?)\s*([^\x00-\x7F]+)\s*([\d\[\-])', line):
        bad_arrow = match.group(2)
        result = result.replace(bad_arrow, arrow)
    return result

# Collect ALL model layers (including static) to preserve index mapping
layer_anims = []
current_anims = {}
in_layer = False
dur = 120
fr = 30

for line in text.split('\n'):
    line = line.strip()
    if line.startswith('anim '):
        for p in line.split():
            if p.startswith('dur='): dur = int(p[4:])
            if p.startswith('fr='): fr = int(p[3:])
    elif line.startswith('layer '):
        if in_layer:
            layer_anims.append(current_anims)
        current_anims = {}
        in_layer = True
    elif in_layer and has_arrow(line) and '{' not in line:
        normalized = normalize_arrows(line)
        prop = normalized.split()[0]
        current_anims[prop] = normalized

if in_layer:
    layer_anims.append(current_anims)

animated = sum(1 for a in layer_anims if a)
print(f"\nModel: {len(layer_anims)} layers, {animated} animated")
for i, a in enumerate(layer_anims):
    print(f"  Layer {i}: {list(a.keys()) if a else 'static'}")

# Step 5: Apply animations to Lottie layers
import re

def parse_keyframes(prop_line, dur):
    """Convert AnimTOON keyframe line to Lottie keyframes."""
    parts = prop_line.strip().split()

    ease = "smooth"
    remaining = ' '.join(parts[1:])
    if 'ease=' in remaining:
        m = re.search(r'ease=(\w+)', remaining)
        if m:
            ease = m.group(1)
        remaining = re.sub(r'\s*ease=\w+', '', remaining)

    kf_pairs = re.findall(r'([\d.]+)' + arrow + r'([\[\d.,\-]+)', remaining)
    if not kf_pairs:
        return None

    keyframes = []
    for time_str, val_str in kf_pairs:
        frame = round(float(time_str) * dur)
        if val_str.startswith('['):
            vals = [float(v) for v in val_str.strip('[]').split(',')]
        else:
            vals = [float(val_str)]

        if ease == "linear":
            ix, iy, ox, oy = [0.167], [0.167], [0.833], [0.833]
        else:
            ix, iy, ox, oy = [0.667], [1], [0.333], [0]

        n = len(vals)
        kf = {"t": frame, "s": vals, "i": {"x": ix * n, "y": iy * n}, "o": {"x": ox * n, "y": oy * n}}
        keyframes.append(kf)

    if keyframes:
        keyframes[-1].pop("i", None)
        keyframes[-1].pop("o", None)

    return {"a": 1, "k": keyframes}


# Get dur/fr from model output or defaults
dur = 120
fr = 30
for line in text.split('\n'):
    if line.strip().startswith('anim '):
        for p in line.strip().split():
            if p.startswith('dur='):
                dur = int(p[4:])
            if p.startswith('fr='):
                fr = int(p[3:])

# Update Lottie timing
lottie_dict['fr'] = fr
lottie_dict['op'] = dur

# Apply animations to matching layers
for i, layer in enumerate(layers):
    layer['op'] = dur
    if i < len(layer_anims):
        anims = layer_anims[i]
        ks = layer.get('ks', {})

        for prop, prop_line in anims.items():
            kf = parse_keyframes(prop_line, dur)
            if not kf:
                continue

            if prop == 'opacity':
                ks['o'] = kf
            elif prop == 'rot':
                ks['r'] = kf
            elif prop == 'scale':
                for k in kf.get('k', []):
                    if 's' in k and len(k['s']) == 2:
                        k['s'].append(100)
                    # Multiply by base scale factor so [100,100] = full canvas size
                    if 's' in k:
                        k['s'] = [round(v * scale_factor / 100, 2) for v in k['s']]
                ks['s'] = kf
            elif prop == 'pos':
                for k in kf.get('k', []):
                    if 's' in k and len(k['s']) >= 2:
                        k['s'] = [round(k['s'][0] * 512, 2), round(k['s'][1] * 512, 2), 0]
                ks['p'] = kf

        layer['ks'] = ks
        print(f"  Applied {list(anims.keys())} to Layer {i}")

# Save as .lottie
import zipfile

lottie_json = json.dumps(lottie_dict, separators=(',', ':'))
manifest = json.dumps({
    "version": "1.0",
    "generator": "AnimTOON-SVG",
    "animations": [{"id": "anim_0", "speed": 1, "loop": True, "autoplay": True}]
}, separators=(',', ':'))

with zipfile.ZipFile('outputs/' + os.path.splitext(os.path.basename(svg_file))[0] + '.lottie', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("manifest.json", manifest)
    zf.writestr("animations/anim_0.json", lottie_json)

# Also save raw JSON for preview
with open('outputs/' + os.path.splitext(os.path.basename(svg_file))[0] + '.json', 'w') as f:
    json.dump(lottie_dict, f, indent=2)

out_name = os.path.splitext(os.path.basename(svg_file))[0]
print(f"\nSaved: outputs/{out_name}.lottie")
print(f"Saved: outputs/{out_name}.json")
print(f"Generation time: {t:.1f}s")
