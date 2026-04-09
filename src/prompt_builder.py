"""
Prompt Builder: Parse SVG and build structured prompts for AnimTOON model.

Takes SVG file → extracts layers, colors, positions, shapes →
builds a detailed "The video shows..." prompt that matches training data format.
"""
import json
import sys
sys.path.insert(0, 'src')

try:
    from lottie import parsers
    HAS_LOTTIE = True
except ImportError:
    HAS_LOTTIE = False

import xml.etree.ElementTree as ET
import re


def hex_to_name(hex_color):
    """Convert hex color to human-readable name."""
    hex_color = hex_color.upper().strip('#')
    if len(hex_color) != 6:
        return hex_color
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

    colors = [
        ((255, 255, 255), "white"), ((0, 0, 0), "black"),
        ((255, 0, 0), "red"), ((0, 255, 0), "green"), ((0, 0, 255), "blue"),
        ((255, 255, 0), "yellow"), ((255, 165, 0), "orange"), ((128, 0, 128), "purple"),
        ((255, 192, 203), "pink"), ((128, 128, 128), "gray"), ((165, 42, 42), "brown"),
        ((0, 255, 255), "cyan"), ((0, 128, 128), "teal"), ((0, 0, 128), "navy"),
    ]

    best_name = hex_color
    best_dist = float('inf')
    for (cr, cg, cb), name in colors:
        dist = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def pos_to_words(cx, cy):
    """Convert normalized position to words."""
    h = "left" if cx < 0.33 else "right" if cx > 0.66 else "center"
    v = "top" if cy < 0.33 else "bottom" if cy > 0.66 else "middle"
    if h == "center" and v == "middle":
        return "in the center"
    if h == "center":
        return f"at the {v}"
    if v == "middle":
        return f"on the {h} side"
    return f"at the {v}-{h}"


def parse_svg_elements(svg_path):
    """Parse SVG and extract element info."""
    tree = ET.parse(svg_path)
    root = tree.getroot()

    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag.split('}')[0] + '}'

    viewbox = root.get('viewBox', '0 0 512 512')
    vb = [float(x) for x in viewbox.split()]
    vb_x, vb_y, vb_w, vb_h = vb[0], vb[1], vb[2], vb[3]

    elements = []

    for elem in root.iter():
        tag = elem.tag.replace(ns, '')
        if tag not in ('path', 'rect', 'circle', 'ellipse', 'polygon', 'line', 'g'):
            continue

        fill = elem.get('fill', '').strip()
        stroke = elem.get('stroke', '').strip()
        d = elem.get('d', '')

        if tag == 'g':
            continue

        if not fill and not stroke:
            continue
        if fill == 'none' and (not stroke or stroke == 'none'):
            continue

        # Estimate position from path data
        cx, cy = 0.5, 0.5
        if d:
            nums = re.findall(r'[-+]?[0-9]*\.?[0-9]+', d)
            if len(nums) >= 4:
                xs = [float(nums[i]) for i in range(0, min(len(nums), 20), 2)]
                ys = [float(nums[i]) for i in range(1, min(len(nums), 20), 2)]
                cx = ((min(xs) + max(xs)) / 2 - vb_x) / vb_w
                cy = ((min(ys) + max(ys)) / 2 - vb_y) / vb_h
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))

        # Estimate shape complexity
        if tag == 'rect':
            shape_type = "rectangle"
        elif tag in ('circle', 'ellipse'):
            shape_type = "circle"
        elif tag == 'path':
            cmds = re.findall(r'[MLHVCSQTAZmlhvcsqtaz]', d)
            if len(cmds) <= 6:
                shape_type = "simple shape"
            elif len(cmds) <= 15:
                shape_type = "detailed shape"
            else:
                shape_type = "complex shape"
        else:
            shape_type = "shape"

        # Size estimate
        size = "small"
        if d:
            nums = re.findall(r'[-+]?[0-9]*\.?[0-9]+', d)
            if len(nums) >= 4:
                xs = [float(n) for n in nums[0::2][:10]]
                ys = [float(n) for n in nums[1::2][:10]]
                w = (max(xs) - min(xs)) / vb_w
                h = (max(ys) - min(ys)) / vb_h
                area = w * h
                if area > 0.3:
                    size = "large"
                elif area > 0.1:
                    size = "medium"

        color = ""
        if fill and fill != 'none':
            color = hex_to_name(fill)
        elif stroke and stroke != 'none':
            color = hex_to_name(stroke) + " outlined"

        elements.append({
            'tag': tag,
            'fill': fill,
            'stroke': stroke,
            'color': color,
            'shape_type': shape_type,
            'size': size,
            'cx': round(cx, 2),
            'cy': round(cy, 2),
            'pos_words': pos_to_words(cx, cy),
        })

    return elements


def build_prompt(svg_path, animation_style="entrance"):
    """Build a structured prompt from SVG file.

    animation_style: 'entrance', 'loop', 'bounce', 'pulse', 'spin'
    """
    elements = parse_svg_elements(svg_path)
    n = len(elements)

    if n == 0:
        return "The video shows a simple animation.", elements

    # Animation templates
    style_templates = {
        "entrance": [
            "scales up from 0 to full size",
            "fades in smoothly",
            "slides in from the side",
            "pops in with a bounce effect",
            "appears with a scale bounce",
        ],
        "loop": [
            "pulses gently by scaling up and down",
            "rotates slowly and continuously",
            "bobs up and down smoothly",
            "breathes with a gentle scale animation",
            "sways side to side",
        ],
        "bounce": [
            "bounces in from above",
            "springs up with elastic motion",
            "drops in and bounces twice",
            "pops in with overshoot",
            "bounces into position",
        ],
        "pulse": [
            "pulses by scaling between 90% and 110%",
            "glows by pulsing opacity",
            "throbs with a heartbeat rhythm",
            "breathes with scale animation",
            "pulses with a smooth rhythm",
        ],
        "spin": [
            "rotates 360 degrees continuously",
            "spins clockwise smoothly",
            "rotates with ease in and out",
            "turns slowly and continuously",
            "spins and scales simultaneously",
        ],
    }

    templates = style_templates.get(animation_style, style_templates["entrance"])

    # Build description
    prompt = f"The video shows an icon animation with {n} layers. "

    for i, elem in enumerate(elements):
        anim_desc = templates[i % len(templates)]

        # Stagger timing for entrance
        if animation_style == "entrance":
            timing = f"at the {'start' if i == 0 else 'same time' if i == 1 else 'end'} of the animation"
        else:
            timing = "throughout the animation"

        prompt += f"Layer {i+1} is a {elem['size']} {elem['color']} {elem['shape_type']} "
        prompt += f"{elem['pos_words']} that {anim_desc} {timing}. "

    prompt += "The animation plays smoothly with clean transitions."

    return prompt, elements


def build_prompt_with_svg_info(svg_path, animation_style="entrance"):
    """Build prompt and also return SVG layer count for validation."""
    prompt, elements = build_prompt(svg_path, animation_style)

    print(f"=== SVG Analysis ===")
    print(f"  Elements found: {len(elements)}")
    for i, e in enumerate(elements):
        print(f"  Layer {i+1}: {e['size']} {e['color']} {e['shape_type']} {e['pos_words']}")
    print()
    print(f"=== Generated Prompt ===")
    print(f"  {prompt[:300]}...")
    print()

    return prompt, elements


if __name__ == "__main__":
    svg_path = sys.argv[1] if len(sys.argv) > 1 else "inputs/robot-svgrepo-com.svg"
    style = sys.argv[2] if len(sys.argv) > 2 else "entrance"

    prompt, elements = build_prompt_with_svg_info(svg_path, style)
