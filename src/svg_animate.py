"""
SVG Animate: Take real SVG paths + AnimTOON model animations → animated .lottie

Pipeline:
  1. Parse SVG → extract real path data, fills, positions
  2. Generate prompt from SVG structure
  3. Model generates AnimTOON animations
  4. Build Lottie JSON using REAL SVG paths + model's animations
  5. Output .lottie file
"""
import json
import re
import os
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_svg_path_to_lottie(d_attr: str, vb_w: float, vb_h: float, vb_x: float = 0, vb_y: float = 0):
    """Convert SVG path 'd' attribute to Lottie bezier shape data.

    Normalizes coordinates relative to viewBox and converts to Lottie format.
    Returns list of Lottie shape dicts (one per sub-path).
    """
    if not d_attr:
        return []

    # Tokenize the path data
    tokens = re.findall(r'[MmLlHhVvCcSsQqTtAaZz]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', d_attr)

    shapes = []
    vertices = []
    in_tangents = []
    out_tangents = []
    current_x, current_y = 0, 0
    start_x, start_y = 0, 0
    i = 0

    def norm_x(x):
        return round((x - vb_x) / vb_w * 512, 2)

    def norm_y(y):
        return round((y - vb_y) / vb_h * 512, 2)

    def save_shape():
        nonlocal vertices, in_tangents, out_tangents
        if len(vertices) >= 2:
            shapes.append({
                "ty": "sh",
                "ks": {
                    "a": 0,
                    "k": {
                        "i": in_tangents[:],
                        "o": out_tangents[:],
                        "v": vertices[:],
                        "c": True
                    }
                },
                "nm": f"Path {len(shapes) + 1}",
                "hd": False
            })
        vertices = []
        in_tangents = []
        out_tangents = []

    while i < len(tokens):
        cmd = tokens[i]

        if cmd in ('M', 'm'):
            if vertices:
                save_shape()
            i += 1
            if i + 1 < len(tokens):
                x, y = float(tokens[i]), float(tokens[i+1])
                if cmd == 'm':
                    x += current_x
                    y += current_y
                current_x, current_y = x, y
                start_x, start_y = x, y
                vertices.append([norm_x(x), norm_y(y)])
                in_tangents.append([0, 0])
                out_tangents.append([0, 0])
                i += 2
                # Implicit lineto after moveto
                while i + 1 < len(tokens) and tokens[i] not in 'MmLlHhVvCcSsQqTtAaZz':
                    x, y = float(tokens[i]), float(tokens[i+1])
                    if cmd == 'm':
                        x += current_x
                        y += current_y
                    current_x, current_y = x, y
                    vertices.append([norm_x(x), norm_y(y)])
                    in_tangents.append([0, 0])
                    out_tangents.append([0, 0])
                    i += 2

        elif cmd in ('L', 'l'):
            i += 1
            while i + 1 < len(tokens) and tokens[i] not in 'MmLlHhVvCcSsQqTtAaZz':
                x, y = float(tokens[i]), float(tokens[i+1])
                if cmd == 'l':
                    x += current_x
                    y += current_y
                current_x, current_y = x, y
                vertices.append([norm_x(x), norm_y(y)])
                in_tangents.append([0, 0])
                out_tangents.append([0, 0])
                i += 2

        elif cmd in ('H', 'h'):
            i += 1
            while i < len(tokens) and tokens[i] not in 'MmLlHhVvCcSsQqTtAaZz':
                x = float(tokens[i])
                if cmd == 'h':
                    x += current_x
                current_x = x
                vertices.append([norm_x(x), norm_y(current_y)])
                in_tangents.append([0, 0])
                out_tangents.append([0, 0])
                i += 1

        elif cmd in ('V', 'v'):
            i += 1
            while i < len(tokens) and tokens[i] not in 'MmLlHhVvCcSsQqTtAaZz':
                y = float(tokens[i])
                if cmd == 'v':
                    y += current_y
                current_y = y
                vertices.append([norm_x(current_x), norm_y(y)])
                in_tangents.append([0, 0])
                out_tangents.append([0, 0])
                i += 1

        elif cmd in ('C', 'c'):
            i += 1
            while i + 5 < len(tokens) and tokens[i] not in 'MmLlHhVvCcSsQqTtAaZz':
                x1, y1 = float(tokens[i]), float(tokens[i+1])
                x2, y2 = float(tokens[i+2]), float(tokens[i+3])
                x, y = float(tokens[i+4]), float(tokens[i+5])
                if cmd == 'c':
                    x1 += current_x; y1 += current_y
                    x2 += current_x; y2 += current_y
                    x += current_x; y += current_y
                # Out tangent of previous point
                if out_tangents:
                    px, py = vertices[-1]
                    out_tangents[-1] = [round(norm_x(x1) - px, 2), round(norm_y(y1) - py, 2)]
                # In tangent of new point
                nx, ny = norm_x(x), norm_y(y)
                in_tan = [round(norm_x(x2) - nx, 2), round(norm_y(y2) - ny, 2)]
                vertices.append([nx, ny])
                in_tangents.append(in_tan)
                out_tangents.append([0, 0])
                current_x, current_y = x, y
                i += 6

        elif cmd in ('S', 's'):
            i += 1
            while i + 3 < len(tokens) and tokens[i] not in 'MmLlHhVvCcSsQqTtAaZz':
                x2, y2 = float(tokens[i]), float(tokens[i+1])
                x, y = float(tokens[i+2]), float(tokens[i+3])
                if cmd == 's':
                    x2 += current_x; y2 += current_y
                    x += current_x; y += current_y
                nx, ny = norm_x(x), norm_y(y)
                in_tan = [round(norm_x(x2) - nx, 2), round(norm_y(y2) - ny, 2)]
                vertices.append([nx, ny])
                in_tangents.append(in_tan)
                out_tangents.append([0, 0])
                current_x, current_y = x, y
                i += 4

        elif cmd in ('Z', 'z'):
            current_x, current_y = start_x, start_y
            save_shape()
            i += 1

        elif cmd in ('A', 'a'):
            # Arc — approximate as line to endpoint
            i += 1
            while i + 6 < len(tokens) and tokens[i] not in 'MmLlHhVvCcSsQqTtAaZz':
                # rx ry x-rotation large-arc sweep x y
                x = float(tokens[i+5])
                y = float(tokens[i+6])
                if cmd == 'a':
                    x += current_x
                    y += current_y
                current_x, current_y = x, y
                vertices.append([norm_x(x), norm_y(y)])
                in_tangents.append([0, 0])
                out_tangents.append([0, 0])
                i += 7
        else:
            i += 1

    if vertices:
        save_shape()

    return shapes


def hex_to_rgba(hex_color: str) -> list:
    """Convert hex color to Lottie RGBA (0-1 range)."""
    hex_color = hex_color.strip('#')
    if len(hex_color) == 3:
        hex_color = ''.join(c*2 for c in hex_color)
    if len(hex_color) != 6:
        return [0.5, 0.5, 0.5, 1]
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255
    return [round(r, 4), round(g, 4), round(b, 4), 1]


def parse_animtoon_animations(animtoon_text: str) -> dict:
    """Parse AnimTOON text and extract animations per layer index."""
    animations = {}
    current_layer = None
    layer_idx = 0
    arrow = "\u2192"

    for line in animtoon_text.split('\n'):
        line = line.strip()
        if not line:
            continue

        if line.startswith('anim '):
            # Parse header
            parts = line.split()
            for p in parts[1:]:
                if p.startswith('fr='):
                    animations['_fr'] = int(p[3:])
                elif p.startswith('dur='):
                    animations['_dur'] = int(p[4:])
            continue

        if line.startswith('layer '):
            current_layer = layer_idx
            animations[current_layer] = {}
            layer_idx += 1
            continue

        if current_layer is not None and arrow in line and '{' not in line:
            prop = line.split()[0]  # rot, scale, opacity, pos
            animations[current_layer][prop] = line

    return animations


def build_keyframes(prop_line: str, dur: int, fr: int, prop_type: str) -> dict:
    """Convert AnimTOON property line to Lottie keyframe dict."""
    arrow = "\u2192"
    parts = prop_line.strip().split()
    prop_name = parts[0]

    # Extract ease
    ease = "smooth"
    remaining = ' '.join(parts[1:])
    if 'ease=' in remaining:
        ease_match = re.search(r'ease=(\w+)', remaining)
        if ease_match:
            ease = ease_match.group(1)
        remaining = re.sub(r'\s*ease=\w+', '', remaining)

    # Parse keyframes: time→value pairs
    kf_pattern = re.findall(r'([\d.]+)' + arrow + r'([\[\d.,\-]+)', remaining)

    if not kf_pattern:
        return None

    keyframes = []
    for time_str, val_str in kf_pattern:
        t = float(time_str)
        frame = round(t * dur)

        if val_str.startswith('['):
            vals = [float(v) for v in val_str.strip('[]').split(',')]
        else:
            vals = [float(val_str)]

        # Easing curves
        if ease == "linear":
            ix, iy = [0.167], [0.167]
            ox, oy = [0.833], [0.833]
        elif ease == "fade":
            ix, iy = [0.667], [1]
            ox, oy = [0.333], [0]
        else:  # smooth
            ix, iy = [0.667], [1]
            ox, oy = [0.333], [0]

        kf = {"t": frame, "s": vals}
        if len(keyframes) > 0 or len(kf_pattern) > 1:
            # Add easing to non-last keyframes
            n = len(vals)
            kf["i"] = {"x": ix * n, "y": iy * n}
            kf["o"] = {"x": ox * n, "y": oy * n}

        keyframes.append(kf)

    # Last keyframe doesn't need easing
    if keyframes:
        last = keyframes[-1]
        last.pop("i", None)
        last.pop("o", None)

    return {"a": 1, "k": keyframes}


def svg_to_animated_lottie(svg_path: str, animtoon_text: str, output_path: str):
    """Combine real SVG paths with AnimTOON animations into .lottie file."""

    # Parse SVG
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag.split('}')[0] + '}'

    viewbox = root.get('viewBox', '0 0 512 512')
    vb_parts = viewbox.split()
    vb_x, vb_y = float(vb_parts[0]), float(vb_parts[1])
    vb_w, vb_h = float(vb_parts[2]), float(vb_parts[3])

    # Parse animations from model output
    anims = parse_animtoon_animations(animtoon_text)
    fr = anims.get('_fr', 30)
    dur = anims.get('_dur', 120)

    # Build Lottie layers from SVG elements
    layers = []
    svg_elements = []

    for elem in root.iter():
        tag = elem.tag.replace(ns, '')
        if tag in ('path', 'rect', 'circle', 'ellipse', 'polygon'):
            svg_elements.append(elem)

    for idx, elem in enumerate(svg_elements):
        tag = elem.tag.replace(ns, '')
        fill = elem.get('fill', '')
        stroke = elem.get('stroke', '')
        stroke_width = elem.get('stroke-width', '0')
        d_attr = elem.get('d', '')

        # Build shape items
        items = []

        # Convert SVG paths to Lottie shapes
        if tag == 'path' and d_attr:
            lottie_shapes = parse_svg_path_to_lottie(d_attr, vb_w, vb_h, vb_x, vb_y)
            items.extend(lottie_shapes)
        elif tag == 'rect':
            x = float(elem.get('x', 0))
            y = float(elem.get('y', 0))
            w = float(elem.get('width', 0))
            h = float(elem.get('height', 0))
            items.append({
                "ty": "rc",
                "s": {"a": 0, "k": [round(w / vb_w * 512, 2), round(h / vb_h * 512, 2)]},
                "p": {"a": 0, "k": [round((x + w/2 - vb_x) / vb_w * 512, 2),
                                     round((y + h/2 - vb_y) / vb_h * 512, 2)]},
                "r": {"a": 0, "k": 0},
                "nm": "Rect 1",
                "hd": False
            })
        elif tag in ('circle', 'ellipse'):
            cx = float(elem.get('cx', 0))
            cy = float(elem.get('cy', 0))
            if tag == 'circle':
                r = float(elem.get('r', 10))
                sx, sy = r * 2, r * 2
            else:
                sx = float(elem.get('rx', 10)) * 2
                sy = float(elem.get('ry', 10)) * 2
            items.append({
                "ty": "el",
                "s": {"a": 0, "k": [round(sx / vb_w * 512, 2), round(sy / vb_h * 512, 2)]},
                "p": {"a": 0, "k": [round((cx - vb_x) / vb_w * 512, 2),
                                     round((cy - vb_y) / vb_h * 512, 2)]},
                "nm": "Ellipse 1",
                "hd": False
            })

        if not items:
            continue

        # Add fill
        if fill and fill.lower() != 'none':
            items.append({
                "ty": "fl",
                "c": {"a": 0, "k": hex_to_rgba(fill)},
                "o": {"a": 0, "k": 100},
                "r": 1, "bm": 0,
                "nm": "Fill 1", "hd": False
            })

        # Add stroke
        if stroke and stroke.lower() != 'none':
            sw = float(stroke_width) if stroke_width else 1
            items.append({
                "ty": "st",
                "c": {"a": 0, "k": hex_to_rgba(stroke)},
                "o": {"a": 0, "k": 100},
                "w": {"a": 0, "k": round(sw / vb_w * 512, 2)},
                "lc": 2, "lj": 2, "bm": 0,
                "nm": "Stroke 1", "hd": False
            })

        # Add transform
        items.append({
            "ty": "tr",
            "p": {"a": 0, "k": [0, 0]},
            "a": {"a": 0, "k": [0, 0]},
            "s": {"a": 0, "k": [100, 100]},
            "r": {"a": 0, "k": 0},
            "o": {"a": 0, "k": 100},
            "sk": {"a": 0, "k": 0},
            "sa": {"a": 0, "k": 0},
            "nm": "Transform"
        })

        # Build layer transform with animations
        layer_anims = anims.get(idx, {})

        # Default transforms
        ks_opacity = {"a": 0, "k": 100}
        ks_rotation = {"a": 0, "k": 0}
        ks_position = {"a": 0, "k": [256, 256, 0]}
        ks_anchor = {"a": 0, "k": [256, 256, 0]}
        ks_scale = {"a": 0, "k": [100, 100, 100]}

        # Apply model's animations
        for prop, prop_line in layer_anims.items():
            kf = build_keyframes(prop_line, dur, fr, prop)
            if kf:
                if prop == 'opacity':
                    ks_opacity = kf
                elif prop == 'rot':
                    ks_rotation = kf
                elif prop == 'scale':
                    # Add z component to scale keyframes
                    for k in kf.get('k', []):
                        if 's' in k and len(k['s']) == 2:
                            k['s'].append(100)
                    ks_scale = kf
                elif prop == 'pos':
                    # Convert normalized pos to pixels
                    for k in kf.get('k', []):
                        if 's' in k:
                            if len(k['s']) >= 2:
                                k['s'] = [round(k['s'][0] * 512, 2), round(k['s'][1] * 512, 2), 0]
                    ks_position = kf

        layer = {
            "ddd": 0,
            "ind": idx + 1,
            "ty": 4,
            "nm": f"SVG_Layer_{idx + 1}",
            "sr": 1,
            "ks": {
                "o": ks_opacity,
                "r": ks_rotation,
                "p": ks_position,
                "a": ks_anchor,
                "s": ks_scale
            },
            "ao": 0,
            "ip": 0,
            "op": dur,
            "st": 0,
            "bm": 0,
            "shapes": [{"ty": "gr", "it": items, "nm": f"Group {idx + 1}", "bm": 0, "hd": False}]
        }

        layers.append(layer)

    # Build full Lottie JSON
    lottie = {
        "v": "5.7.4",
        "fr": fr,
        "ip": 0,
        "op": dur,
        "w": 512,
        "h": 512,
        "nm": "SVG_Animated",
        "ddd": 0,
        "assets": [],
        "layers": layers,
        "markers": []
    }

    # Pack as .lottie (ZIP)
    lottie_json = json.dumps(lottie, separators=(',', ':'))
    manifest = json.dumps({
        "version": "1.0",
        "generator": "AnimTOON-SVG",
        "animations": [{"id": "anim_0", "speed": 1, "loop": True, "autoplay": True}]
    }, separators=(',', ':'))

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", manifest)
        zf.writestr("animations/anim_0.json", lottie_json)

    return len(layers)


if __name__ == "__main__":
    svg_path = sys.argv[1] if len(sys.argv) > 1 else "inputs/robot-svgrepo-com.svg"

    # Default test animation
    test_anim = """anim fr=30 dur=120

layer body shape
  scale 0.0\u2192[0,0] 0.1\u2192[110,110] 0.2\u2192[100,100] ease=smooth
  opacity 0.0\u21920 0.1\u2192100 ease=fade

layer ears shape
  scale 0.1\u2192[0,0] 0.2\u2192[110,110] 0.3\u2192[100,100] ease=smooth

layer chest shape
  scale 0.2\u2192[0,0] 0.3\u2192[110,110] 0.4\u2192[100,100] ease=smooth

layer outline shape
  scale 0.0\u2192[0,0] 0.15\u2192[105,105] 0.3\u2192[100,100] ease=smooth

layer details shape
  scale 0.15\u2192[0,0] 0.3\u2192[110,110] 0.4\u2192[100,100] ease=smooth
  opacity 0.0\u21920 0.2\u2192100 ease=fade
"""

    n = svg_to_animated_lottie(svg_path, test_anim, "outputs/robot_real_svg.lottie")
    print(f"Saved: outputs/robot_real_svg.lottie ({n} layers with real SVG paths)")
