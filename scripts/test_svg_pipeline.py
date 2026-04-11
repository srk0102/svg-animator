"""Test full SVG -> animated Lottie pipeline.
Applies model animations to individual shape groups within SVG layers.
"""
import sys
import os
import json
import copy
import re
import time

sys.path.insert(0, 'src')

# ============================================================
# Step 1: Convert SVG to Lottie using python-lottie
# ============================================================
from lottie import parsers
import xml.etree.ElementTree as ET

svg_file = sys.argv[1] if len(sys.argv) > 1 else 'inputs/apple.svg'
print(f"SVG: {svg_file}")

anim = parsers.svg.parse_svg_file(svg_file)

# Get viewBox for scaling
tree = ET.parse(svg_file)
root = tree.getroot()
ns = ''
if root.tag.startswith('{'):
    ns = root.tag.split('}')[0] + '}'
viewbox = root.get('viewBox', '0 0 512 512')
vb = [float(x) for x in viewbox.split()]
vb_x, vb_y, vb_w, vb_h = vb[0], vb[1], vb[2], vb[3]
scale_factor = 512 / max(vb_w, vb_h) * 100

# Set native size for path alignment
anim.width = vb_w
anim.height = vb_h

lottie_dict = anim.to_dict()

# Scale canvas to 512x512
lottie_dict['w'] = 512
lottie_dict['h'] = 512

# Scale layer transform to fit canvas (this stays static)
for layer in lottie_dict.get('layers', []):
    ks = layer.get('ks', {})
    ks['s'] = {"a": 0, "k": [scale_factor, scale_factor, 100]}
    ks['p'] = {"a": 0, "k": [256, 256, 0]}
    ks['a'] = {"a": 0, "k": [vb_w / 2, vb_h / 2, 0]}
    layer['ks'] = ks

# Count shape groups across all layers
layers = lottie_dict.get('layers', [])
all_shape_groups = []
for layer in layers:
    for sg in layer.get('shapes', []):
        all_shape_groups.append((layer, sg))

shape_count = len(all_shape_groups)
print(f"SVG: {len(layers)} layers, {shape_count} shape groups")

# ============================================================
# Step 2: Build prompt
# ============================================================
from prompt_builder import build_prompt_with_svg_info
_, elements = build_prompt_with_svg_info(svg_file, 'entrance')

if shape_count > 20:
    prompt = (
        f"A character with {shape_count} parts performing an idle animation. "
        "Animate key parts only: eyes blink (scale Y 100->10->100), "
        "arms/claws rotate gently (rot +/-15), body breathes (scale 100->102->100), "
        "head bobs (pos Y shifts). Keep other parts static. "
        "Use sparse keyframes (3-5 per property). "
        f"Generate animation for {min(shape_count, 15)} most important parts."
    )
elif shape_count > 3:
    prompt = (
        f"A character with {shape_count} body parts performing an idle animation. "
        "Use sparse keyframes (3-5 per property). "
        "Animate all parts with coordinated subtle movement, head bobbing, and body sway."
    )
else:
    prompt = (
        "A shape performing a smooth bounce entrance and gentle pulse animation. "
        "Use sparse keyframes (3-5 per property)."
    )

print(f"\nPrompt: {prompt[:200]}...")

# ============================================================
# Step 3: Generate animation with model
# ============================================================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model...")
m_tokenizer = AutoTokenizer.from_pretrained('models/animtoon-3b-v3-merged')
m_model = AutoModelForCausalLM.from_pretrained(
    'models/animtoon-3b-v3-merged', dtype=torch.float16, device_map='cuda'
)

messages = [{'role': 'user', 'content': f'Generate AnimTOON animation: {prompt}'}]
text_input = m_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = m_tokenizer(text_input, return_tensors='pt').to('cuda')

start = time.time()
with torch.no_grad():
    out = m_model.generate(**inputs, max_new_tokens=1024, do_sample=False)
t = time.time() - start

text = m_model = None  # free GPU memory
text_out = m_tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
gen_tokens = out[0].shape[0] - inputs['input_ids'].shape[1]
del m_model, out, inputs
torch.cuda.empty_cache()

print(f"\nMODEL OUTPUT ({gen_tokens} tokens, {t:.1f}s):")
print(text_out)

# ============================================================
# Step 4: Parse model animations
# ============================================================
arrow = "\u2192"


def has_arrow(line):
    if arrow in line:
        return True
    return bool(re.search(r'\d[^\x00-\x7F]+[\d\[\-]', line))


def normalize_arrows(line):
    result = line
    for match in re.finditer(r'(\d(?:\.\d+)?)\s*([^\x00-\x7F]+)\s*([\d\[\-])', line):
        result = result.replace(match.group(2), arrow)
    return result


# Parse all model layers
model_layers = []
current = {}
in_layer = False
dur = 120
fr = 30

for line in text_out.split('\n'):
    line = line.strip()
    if line.startswith('anim '):
        for p in line.split():
            if p.startswith('dur='): dur = int(p[4:])
            if p.startswith('fr='): fr = int(p[3:])
    elif line.startswith('layer'):
        if in_layer:
            model_layers.append(current)
        current = {}
        in_layer = True
    elif in_layer and has_arrow(line) and '{' not in line:
        normalized = normalize_arrows(line)
        prop = normalized.split()[0]
        current[prop] = normalized

if in_layer:
    model_layers.append(current)

# Only keep animated layers
animated_layers = [a for a in model_layers if a]
print(f"\nModel: {len(model_layers)} layers, {len(animated_layers)} animated")
for i, a in enumerate(model_layers):
    print(f"  Layer {i}: {list(a.keys()) if a else 'static'}")


# ============================================================
# Step 5: Build keyframes
# ============================================================
def parse_keyframes(prop_line, duration):
    """Convert AnimTOON property line to Lottie keyframe dict."""
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
        frame = round(float(time_str) * duration)
        if val_str.startswith('['):
            vals = [float(v) for v in val_str.strip('[]').split(',')]
        else:
            vals = [float(val_str)]

        if ease == "linear":
            ix, iy, ox, oy = [0.167], [0.167], [0.833], [0.833]
        else:
            ix, iy, ox, oy = [0.667], [1], [0.333], [0]

        n = len(vals)
        kf = {
            "t": frame, "s": vals,
            "i": {"x": ix * n, "y": iy * n},
            "o": {"x": ox * n, "y": oy * n}
        }
        keyframes.append(kf)

    if keyframes:
        keyframes[-1].pop("i", None)
        keyframes[-1].pop("o", None)

    return {"a": 1, "k": keyframes}


# ============================================================
# Step 6: Apply animations to shape group transforms
# ============================================================
lottie_dict['fr'] = fr
lottie_dict['op'] = dur

for layer in layers:
    layer['op'] = dur


def get_shape_bbox(items):
    """Calculate bounding box center of a shape group's path data.
    Returns (cx, cy) in the shape's local coordinate space."""
    all_x = []
    all_y = []

    for item in items:
        ty = item.get('ty')

        if ty == 'sh':
            # Bezier shape — extract vertices
            ks = item.get('ks', {})
            k = ks.get('k', {})
            if isinstance(k, dict):
                verts = k.get('v', [])
                for v in verts:
                    if isinstance(v, list) and len(v) >= 2:
                        all_x.append(v[0])
                        all_y.append(v[1])

        elif ty == 'el':
            # Ellipse — center is at position, size defines bounds
            p = item.get('p', {}).get('k', [0, 0])
            s = item.get('s', {}).get('k', [10, 10])
            if isinstance(p, list) and len(p) >= 2:
                if isinstance(s, list) and len(s) >= 2:
                    all_x.extend([p[0] - s[0]/2, p[0] + s[0]/2])
                    all_y.extend([p[1] - s[1]/2, p[1] + s[1]/2])

        elif ty == 'rc':
            # Rectangle
            p = item.get('p', {}).get('k', [0, 0])
            s = item.get('s', {}).get('k', [10, 10])
            if isinstance(p, list) and len(p) >= 2:
                if isinstance(s, list) and len(s) >= 2:
                    all_x.extend([p[0] - s[0]/2, p[0] + s[0]/2])
                    all_y.extend([p[1] - s[1]/2, p[1] + s[1]/2])

        elif ty == 'gr':
            # Nested group — recurse
            sub_items = item.get('it', [])
            sub_cx, sub_cy = get_shape_bbox(sub_items)
            if sub_cx is not None:
                all_x.append(sub_cx)
                all_y.append(sub_cy)

    if all_x and all_y:
        cx = (min(all_x) + max(all_x)) / 2
        cy = (min(all_y) + max(all_y)) / 2
        return round(cx, 2), round(cy, 2)

    return None, None


if not animated_layers:
    print("\nNo animations to apply.")
else:
    applied_count = 0

    for layer in layers:
        shape_groups = layer.get('shapes', [])

        for sg_idx, sg in enumerate(shape_groups):
            items = sg.get('it', [])
            tr = None
            tr_idx = None

            for idx, item in enumerate(items):
                if item.get('ty') == 'tr':
                    tr = item
                    tr_idx = idx
                    break

            if tr is None:
                continue

            # Pick animation: first N get direct, rest cycle every 3rd
            if sg_idx < len(animated_layers):
                anim_data = animated_layers[sg_idx]
            elif animated_layers:
                if sg_idx % 3 == 0:
                    anim_data = animated_layers[sg_idx % len(animated_layers)]
                else:
                    continue
            else:
                continue

            if not anim_data:
                continue

            # Only rot, scale, opacity — NEVER position
            safe_props = {k: v for k, v in anim_data.items()
                          if k in ('rot', 'scale', 'opacity')}
            if not safe_props:
                continue

            # ── FIX: Set anchor point to BBox center ──
            # This makes rotation/scale happen around the shape's center
            # instead of (0,0) which causes parts to fly off
            cx, cy = get_shape_bbox(items)
            if cx is not None and cy is not None:
                # Set anchor to shape center
                tr['a'] = {"a": 0, "k": [cx, cy]}
                # Offset position by same amount so shape stays in place
                old_p = tr.get('p', {}).get('k', [0, 0])
                if isinstance(old_p, list) and len(old_p) >= 2:
                    tr['p'] = {"a": 0, "k": [old_p[0] + cx, old_p[1] + cy]}
                else:
                    tr['p'] = {"a": 0, "k": [cx, cy]}

            # Apply rotation
            if 'rot' in safe_props:
                kf = parse_keyframes(safe_props['rot'], dur)
                if kf:
                    tr['r'] = kf

            # Apply scale (relative to shape, not canvas)
            if 'scale' in safe_props:
                kf = parse_keyframes(safe_props['scale'], dur)
                if kf:
                    for k in kf.get('k', []):
                        if 's' in k and len(k['s']) == 2:
                            k['s'].append(100)
                    tr['s'] = kf

            # Apply opacity
            if 'opacity' in safe_props:
                kf = parse_keyframes(safe_props['opacity'], dur)
                if kf:
                    tr['o'] = kf

            items[tr_idx] = tr
            applied_count += 1
            print(f"  Shape {sg_idx}: anchor=({cx},{cy}) anim={list(safe_props.keys())}")

    print(f"\nTotal: {applied_count} shape groups animated")

# ============================================================
# Step 7: Save output
# ============================================================
import zipfile

out_name = os.path.splitext(os.path.basename(svg_file))[0]

lottie_json = json.dumps(lottie_dict, separators=(',', ':'))
manifest = json.dumps({
    "version": "1.0",
    "generator": "AnimTOON-SVG",
    "animations": [{"id": "anim_0", "speed": 1, "loop": True, "autoplay": True}]
}, separators=(',', ':'))

with zipfile.ZipFile(f'outputs/{out_name}.lottie', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("manifest.json", manifest)
    zf.writestr("animations/anim_0.json", lottie_json)

with open(f'outputs/{out_name}.json', 'w') as f:
    json.dump(lottie_dict, f, indent=2)

print(f"\nSaved: outputs/{out_name}.lottie")
print(f"Saved: outputs/{out_name}.json")
print(f"Generation time: {t:.1f}s")
