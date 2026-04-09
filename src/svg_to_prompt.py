"""Parse SVG file and generate a description for AnimTOON model."""
import re
import xml.etree.ElementTree as ET


def parse_svg_layers(svg_path: str) -> list:
    """Extract layers/elements from SVG with their properties."""
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Handle SVG namespace
    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag.split('}')[0] + '}'

    # Get viewBox for coordinate normalization
    viewbox = root.get('viewBox', '0 0 512 512')
    vb_parts = viewbox.split()
    vb_x, vb_y = float(vb_parts[0]), float(vb_parts[1])
    vb_w, vb_h = float(vb_parts[2]), float(vb_parts[3])

    layers = []

    def color_name(hex_color):
        """Approximate color name from hex."""
        hex_color = hex_color.upper().strip('#')
        if len(hex_color) != 6:
            return hex_color
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        if r > 200 and g > 200 and b > 200: return "white"
        if r < 50 and g < 50 and b < 50: return "dark gray"
        if r > 200 and g < 100 and b < 100: return "red"
        if r < 100 and g > 150 and b < 100: return "green"
        if r < 100 and g < 100 and b > 200: return "blue"
        if r > 200 and g > 200 and b < 100: return "yellow"
        if r > 200 and g > 100 and b < 100: return "orange"
        if r > 100 and g < 100 and b > 100: return "purple"
        if r > 50 and g > 80 and b > 150: return "blue"
        if r > 150 and g > 80 and b < 100: return "brown"
        return f"#{hex_color}"

    def estimate_shape(d_attr):
        """Guess shape type from path data."""
        if not d_attr:
            return "shape"
        # Count path commands
        commands = re.findall(r'[MLHVCSQTAZmlhvcsqtaz]', d_attr)
        if len(commands) <= 5:
            return "simple shape"
        elif len(commands) <= 15:
            return "medium shape"
        else:
            return "complex shape"

    def get_bbox_center(d_attr):
        """Estimate center position from path data."""
        numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', d_attr or '')
        if len(numbers) < 2:
            return 0.5, 0.5
        nums = [float(n) for n in numbers[:20]]  # first 20 numbers
        xs = nums[0::2]  # even indices
        ys = nums[1::2]  # odd indices
        if not xs or not ys:
            return 0.5, 0.5
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        # Normalize to 0-1
        nx = (cx - vb_x) / vb_w
        ny = (cy - vb_y) / vb_h
        return round(max(0, min(1, nx)), 2), round(max(0, min(1, ny)), 2)

    # Parse all path/rect/circle/ellipse elements
    for i, elem in enumerate(root.iter()):
        tag = elem.tag.replace(ns, '')

        if tag in ('path', 'rect', 'circle', 'ellipse', 'polygon', 'line'):
            fill = elem.get('fill', '')
            stroke = elem.get('stroke', '')
            d = elem.get('d', '')

            # Skip elements with no visual
            if fill == 'none' and stroke == 'none':
                continue

            # Get position
            if tag == 'path':
                cx, cy = get_bbox_center(d)
                shape = estimate_shape(d)
            elif tag == 'rect':
                x = float(elem.get('x', 0))
                y = float(elem.get('y', 0))
                w = float(elem.get('width', 0))
                h = float(elem.get('height', 0))
                cx = round((x + w/2 - vb_x) / vb_w, 2)
                cy = round((y + h/2 - vb_y) / vb_h, 2)
                shape = "rectangle"
            elif tag in ('circle', 'ellipse'):
                cx = float(elem.get('cx', 0))
                cy = float(elem.get('cy', 0))
                cx = round((cx - vb_x) / vb_w, 2)
                cy = round((cy - vb_y) / vb_h, 2)
                shape = "circle"
            else:
                cx, cy = 0.5, 0.5
                shape = tag

            layer = {
                'index': i,
                'tag': tag,
                'shape': shape,
                'fill': fill if fill and fill != 'none' else None,
                'stroke': stroke if stroke and stroke != 'none' else None,
                'color_name': color_name(fill) if fill and fill != 'none' else color_name(stroke) if stroke else 'unknown',
                'cx': cx,
                'cy': cy,
            }
            layers.append(layer)

    return layers


def layers_to_description(layers: list) -> str:
    """Convert parsed layers into a natural language description for the model."""
    n = len(layers)

    # Build position descriptions
    def pos_desc(cx, cy):
        h = "left" if cx < 0.35 else "right" if cx > 0.65 else "center"
        v = "top" if cy < 0.35 else "bottom" if cy > 0.65 else "middle"
        if h == "center" and v == "middle":
            return "in the center"
        return f"at the {v}-{h}"

    parts = []
    for i, layer in enumerate(layers):
        color = layer['color_name']
        shape = layer['shape']
        pos = pos_desc(layer['cx'], layer['cy'])
        parts.append(f"Layer {i+1} is a {color} {shape} {pos}")

    desc = f"The video shows an animation with {n} layers. "
    desc += ". ".join(parts) + ". "
    desc += "The animation has smooth transitions with elements fading in, scaling up, and moving gently."

    return desc


def svg_to_prompt(svg_path: str) -> str:
    """Full pipeline: SVG file → model prompt."""
    layers = parse_svg_layers(svg_path)
    desc = layers_to_description(layers)
    return desc, layers


if __name__ == "__main__":
    import sys
    svg_path = sys.argv[1] if len(sys.argv) > 1 else "inputs/robot-svgrepo-com.svg"
    desc, layers = svg_to_prompt(svg_path)
    print(f"Found {len(layers)} layers:")
    for l in layers:
        print(f"  [{l['index']}] {l['color_name']} {l['shape']} at ({l['cx']}, {l['cy']}) fill={l['fill']} stroke={l['stroke']}")
    print()
    print("Generated prompt:")
    print(desc)
