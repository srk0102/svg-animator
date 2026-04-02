"""
AnimTOON - Compact Lottie Animation Format Converter

Converts Lottie JSON (~70K tokens) to AnimTOON text (~500-800 tokens)
and back. Designed for training smaller/faster animation generation models.

Pipeline: .lottie → Lottie JSON → AnimTOON → Lottie JSON → .lottie
"""

import json
import zipfile
import re
import os
import io
from typing import Optional


# ─── DOTLOTTIE HELPERS ───────────────────────────────────────────────

def dotlottie_to_lottie(dotlottie_path: str) -> tuple:
    """Extract Lottie JSON and manifest from dotLottie ZIP file."""
    with zipfile.ZipFile(dotlottie_path, 'r') as z:
        manifest = json.loads(z.read('manifest.json').decode('utf-8'))
        # Find the animation JSON (usually in animations/ folder)
        anim_files = [f for f in z.namelist() if f.endswith('.json') and f != 'manifest.json']
        if not anim_files:
            raise ValueError("No animation JSON found in dotLottie file")
        anim_name = anim_files[0]
        lottie_json = json.loads(z.read(anim_name).decode('utf-8'))
    return lottie_json, manifest, anim_name


def lottie_to_dotlottie(lottie_json: dict, manifest: dict, anim_name: str, output_path: str):
    """Pack Lottie JSON back into dotLottie ZIP file."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json', json.dumps(manifest))
        z.writestr(anim_name, json.dumps(lottie_json))


# ─── LOTTIE → ANIMTOON ──────────────────────────────────────────────

LAYER_TYPE_MAP = {0: 'precomp', 1: 'solid', 2: 'image', 3: 'null', 4: 'shape', 5: 'text'}


class LottieToAnimTOON:
    """Converts Lottie JSON to compact AnimTOON text format."""

    def convert(self, lottie_json: dict) -> str:
        self.w = lottie_json.get('w', 512)
        self.h = lottie_json.get('h', 512)
        self.fr = lottie_json.get('fr', 60)
        self.ip = lottie_json.get('ip', 0)
        self.op = lottie_json.get('op', 360)
        self.total_frames = self.op - self.ip
        if self.total_frames <= 0:
            self.total_frames = 1

        lines = []
        lines.append(f"anim fr={int(self.fr)} dur={int(self.total_frames)}")
        lines.append("")

        for layer in lottie_json.get('layers', []):
            try:
                layer_lines = self._convert_layer(layer)
                if layer_lines:
                    lines.extend(layer_lines)
                    lines.append("")
            except Exception:
                # Skip problematic layers gracefully
                continue

        return '\n'.join(lines).strip() + '\n'

    def _convert_layer(self, layer: dict) -> list:
        nm = layer.get('nm', 'unnamed')
        # Sanitize name: remove spaces, keep short
        nm_clean = re.sub(r'[^a-zA-Z0-9_]', '', nm)
        ty_num = layer.get('ty', 4)
        ty_str = LAYER_TYPE_MAP.get(ty_num, 'shape')

        lines = [f"layer {nm_clean} {ty_str}"]

        # Extract fill/stroke/path from shapes
        if 'shapes' in layer:
            self._extract_shapes(layer['shapes'], lines)

        # Transform properties (ks)
        ks = layer.get('ks', {})

        # Position (always include)
        pos_line = self._convert_property_pos(ks.get('p', {}))
        if pos_line:
            lines.append(f"  {pos_line}")

        # Rotation (skip if static 0)
        rot_line = self._convert_property_scalar(ks.get('r', {}), 'rot', default=0)
        if rot_line:
            lines.append(f"  {rot_line}")

        # Scale (skip if static [100,100])
        scale_line = self._convert_property_scale(ks.get('s', {}))
        if scale_line:
            lines.append(f"  {scale_line}")

        # Opacity (skip if static 100)
        opa_line = self._convert_property_scalar(ks.get('o', {}), 'opacity', default=100)
        if opa_line:
            lines.append(f"  {opa_line}")

        return lines

    def _extract_shapes(self, shapes: list, lines: list):
        """Recursively extract fill, stroke, path info from shape groups.
        Deduplicates: only unique fill colors, unique strokes, and path count."""
        fills = set()
        strokes = set()
        path_types = {}  # type -> count

        self._collect_shapes(shapes, fills, strokes, path_types)

        # Emit unique fills
        for color in sorted(fills):
            lines.append(f"  fill {color}")
        # Emit unique strokes
        for st in sorted(strokes):
            lines.append(f"  stroke {st}")
        # Emit paths compactly: "path sh x5" if multiple
        for ptype, count in sorted(path_types.items()):
            if count > 1:
                lines.append(f"  path {ptype} x{count}")
            else:
                lines.append(f"  path {ptype}")

    def _collect_shapes(self, shapes: list, fills: set, strokes: set, path_types: dict):
        """Recursively collect unique fills/strokes/paths from shape tree."""
        for shape in shapes:
            ty = shape.get('ty')
            try:
                if ty == 'fl':
                    color = self._extract_color(shape.get('c', {}))
                    if color:
                        fills.add(color)
                elif ty == 'st':
                    color = self._extract_color(shape.get('c', {}))
                    w_val = self._extract_static_value(shape.get('w', {}))
                    if color:
                        w_norm = round(w_val / self.w, 4) if w_val else 0.002
                        strokes.add(f"{color} w={w_norm}")
                elif ty == 'sh':
                    path_types['sh'] = path_types.get('sh', 0) + 1
                elif ty == 'el':
                    size = self._extract_static_value(shape.get('s', {}))
                    if isinstance(size, list) and len(size) >= 2:
                        wn = round(size[0] / self.w, 4)
                        hn = round(size[1] / self.h, 4)
                        key = f"ellipse w={wn} h={hn}"
                    else:
                        key = "ellipse w=0.1 h=0.1"
                    path_types[key] = path_types.get(key, 0) + 1
                elif ty == 'rc':
                    size = self._extract_static_value(shape.get('s', {}))
                    r_val = self._extract_static_value(shape.get('r', {}))
                    if isinstance(size, list) and len(size) >= 2:
                        wn = round(size[0] / self.w, 4)
                        hn = round(size[1] / self.h, 4)
                        rn = round((r_val or 0) / self.w, 4)
                        key = f"rect w={wn} h={hn} r={rn}"
                    else:
                        key = "rect w=0.1 h=0.1 r=0"
                    path_types[key] = path_types.get(key, 0) + 1
                elif ty == 'gr':
                    self._collect_shapes(shape.get('it', []), fills, strokes, path_types)
            except Exception:
                continue

    def _extract_color(self, color_prop: dict) -> Optional[str]:
        """Convert Lottie color [r,g,b,a] (0-1 floats) to #RRGGBB hex."""
        try:
            k = color_prop.get('k', [0, 0, 0])
            if color_prop.get('a', 0) == 1:
                # Animated color - take first keyframe
                if isinstance(k, list) and len(k) > 0 and isinstance(k[0], dict):
                    k = k[0].get('s', [0, 0, 0])
            if isinstance(k, list) and len(k) >= 3:
                r = int(min(max(k[0], 0), 1) * 255)
                g = int(min(max(k[1], 0), 1) * 255)
                b = int(min(max(k[2], 0), 1) * 255)
                return f"#{r:02X}{g:02X}{b:02X}"
        except Exception:
            pass
        return None

    def _extract_static_value(self, prop: dict):
        """Extract static value from a Lottie property."""
        if not prop:
            return None
        k = prop.get('k')
        if prop.get('a', 0) == 1 and isinstance(k, list) and len(k) > 0 and isinstance(k[0], dict):
            return k[0].get('s', k)
        return k

    def _convert_property_pos(self, prop: dict) -> Optional[str]:
        """Convert position property to AnimTOON format."""
        if not prop:
            return None
        animated = prop.get('a', 0) == 1
        k = prop.get('k', [0, 0])

        if animated and isinstance(k, list) and len(k) > 0 and isinstance(k[0], dict):
            kf_parts = []
            for kf in k:
                t = kf.get('t', 0)
                t_norm = round((t - self.ip) / self.total_frames, 2)
                t_norm = max(0.0, min(1.0, t_norm))
                s = kf.get('s', [0, 0, 0])
                if isinstance(s, list) and len(s) >= 2:
                    x = round(s[0] / self.w, 2)
                    y = round(s[1] / self.h, 2)
                    kf_parts.append(f"{t_norm}\u2192[{x},{y}]")
            ease = self._detect_ease(k)
            return f"pos {' '.join(kf_parts)} ease={ease}"
        else:
            # Static position
            if isinstance(k, list) and len(k) >= 2:
                x = round(k[0] / self.w, 2)
                y = round(k[1] / self.h, 2)
                return f"pos [{x},{y}]"
            return None

    def _convert_property_scalar(self, prop: dict, name: str, default=0) -> Optional[str]:
        """Convert scalar property (rotation, opacity) to AnimTOON."""
        if not prop:
            return None
        animated = prop.get('a', 0) == 1
        k = prop.get('k', default)

        if animated and isinstance(k, list) and len(k) > 0 and isinstance(k[0], dict):
            kf_parts = []
            for kf in k:
                t = kf.get('t', 0)
                t_norm = round((t - self.ip) / self.total_frames, 2)
                t_norm = max(0.0, min(1.0, t_norm))
                s = kf.get('s', [0])
                val = s[0] if isinstance(s, list) and len(s) > 0 else s
                val = round(val) if isinstance(val, float) and val == int(val) else (round(val, 1) if isinstance(val, float) else val)
                kf_parts.append(f"{t_norm}\u2192{val}")
            ease = self._detect_ease(k)
            if name == 'opacity':
                ease = 'fade'
            return f"{name} {' '.join(kf_parts)} ease={ease}"
        else:
            # Static - skip if default
            val = k
            if isinstance(val, list):
                val = val[0] if len(val) > 0 else default
            if isinstance(val, float):
                val = round(val) if val == int(val) else round(val, 1)
            if val == default:
                return None
            return f"{name} {val}"

    def _convert_property_scale(self, prop: dict) -> Optional[str]:
        """Convert scale property to AnimTOON."""
        if not prop:
            return None
        animated = prop.get('a', 0) == 1
        k = prop.get('k', [100, 100])

        if animated and isinstance(k, list) and len(k) > 0 and isinstance(k[0], dict):
            kf_parts = []
            for kf in k:
                t = kf.get('t', 0)
                t_norm = round((t - self.ip) / self.total_frames, 2)
                t_norm = max(0.0, min(1.0, t_norm))
                s = kf.get('s', [100, 100])
                sx = round(s[0]) if len(s) > 0 else 100
                sy = round(s[1]) if len(s) > 1 else 100
                kf_parts.append(f"{t_norm}\u2192[{sx},{sy}]")
            ease = self._detect_ease(k)
            return f"scale {' '.join(kf_parts)} ease={ease}"
        else:
            # Static - skip if [100,100]
            if isinstance(k, list) and len(k) >= 2:
                sx, sy = round(k[0]), round(k[1])
                if sx == 100 and sy == 100:
                    return None
                return f"scale [{sx},{sy}]"
            return None

    def _detect_ease(self, keyframes: list) -> str:
        """Detect easing type from Lottie keyframes."""
        if not keyframes or not isinstance(keyframes, list):
            return 'linear'
        for kf in keyframes:
            if not isinstance(kf, dict):
                continue
            i = kf.get('i', {})
            o = kf.get('o', {})
            if not i or not o:
                continue
            # Check for bounce (overshoot) - y values > 1
            iy = i.get('y', [1])
            if isinstance(iy, list):
                if any(v > 1.0 for v in iy if isinstance(v, (int, float))):
                    return 'bounce'
            elif isinstance(iy, (int, float)) and iy > 1.0:
                return 'bounce'
            # Check for linear (x==y on both in/out)
            ix = i.get('x', [0])
            ox = o.get('x', [0])
            oy = o.get('y', [0])
            if ix == iy and ox == oy:
                return 'linear'
            # Otherwise it's smooth (bezier)
            return 'smooth'
        return 'linear'


# ─── ANIMTOON → LOTTIE ──────────────────────────────────────────────

LAYER_TYPE_NUM = {'precomp': 0, 'solid': 1, 'image': 2, 'null': 3, 'shape': 4, 'text': 5}


class AnimTOONToLottie:
    """Converts AnimTOON text back to valid renderable Lottie JSON."""

    def convert(self, animtoon_text: str) -> dict:
        lines = animtoon_text.strip().split('\n')
        header = self._parse_header(lines)

        self.fr = header['fr']
        self.dur = header['dur']
        self.w = 512
        self.h = 512

        layers = self._parse_layers(lines)

        lottie = {
            "v": "5.7.4",
            "fr": self.fr,
            "ip": 0,
            "op": self.dur,
            "w": self.w,
            "h": self.h,
            "nm": "AnimTOON_output",
            "ddd": 0,
            "assets": [],
            "layers": layers,
            "markers": []
        }
        return lottie

    def _parse_header(self, lines: list) -> dict:
        for line in lines:
            line = line.strip()
            if line.startswith('anim '):
                fr_match = re.search(r'fr=(\d+)', line)
                dur_match = re.search(r'dur=(\d+)', line)
                return {
                    'fr': int(fr_match.group(1)) if fr_match else 60,
                    'dur': int(dur_match.group(1)) if dur_match else 360
                }
        return {'fr': 60, 'dur': 360}

    def _parse_layers(self, lines: list) -> list:
        layers = []
        current_layer = None
        current_props = []

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            if stripped.startswith('anim '):
                continue

            if stripped.startswith('layer '):
                # Save previous layer
                if current_layer is not None:
                    layers.append(self._build_layer(current_layer, current_props, len(layers)))
                parts = stripped.split()
                nm = parts[1] if len(parts) > 1 else 'unnamed'
                ty = parts[2] if len(parts) > 2 else 'shape'
                current_layer = {'nm': nm, 'ty': ty}
                current_props = []
            elif current_layer is not None:
                current_props.append(stripped)

        # Don't forget last layer
        if current_layer is not None:
            layers.append(self._build_layer(current_layer, current_props, len(layers)))

        return layers

    def _build_layer(self, layer_info: dict, props: list, index: int) -> dict:
        nm = layer_info['nm']
        ty = LAYER_TYPE_NUM.get(layer_info['ty'], 4)

        fills = []
        strokes = []
        paths = []
        pos_data = None
        rot_data = None
        scale_data = None
        opacity_data = None

        for prop in props:
            try:
                if prop.startswith('fill '):
                    fills.append(prop[5:].strip())
                elif prop.startswith('stroke '):
                    strokes.append(prop[7:].strip())
                elif prop.startswith('path '):
                    path_str = prop[5:].strip()
                    # Handle "path sh x5" → 5 copies
                    count_match = re.search(r'\sx(\d+)$', path_str)
                    if count_match:
                        count = int(count_match.group(1))
                        path_base = path_str[:count_match.start()]
                        paths.extend([path_base] * count)
                    else:
                        paths.append(path_str)
                elif prop.startswith('pos '):
                    val = prop[4:].strip()
                    if '{' not in val:  # skip raw JSON blobs
                        pos_data = val
                elif prop.startswith('rot '):
                    val = prop[4:].strip()
                    if '{' not in val:
                        rot_data = val
                elif prop.startswith('scale '):
                    val = prop[6:].strip()
                    if '{' not in val:
                        scale_data = val
                elif prop.startswith('opacity '):
                    val = prop[8:].strip()
                    if '{' not in val:
                        opacity_data = val
            except Exception:
                continue

        # Build transform (ks)
        ks = {}
        ks['o'] = self._build_scalar_prop(opacity_data, 100)
        ks['r'] = self._build_scalar_prop(rot_data, 0)
        ks['p'] = self._build_pos_prop(pos_data)
        ks['a'] = self._build_anchor_from_pos(pos_data)
        ks['s'] = self._build_scale_prop(scale_data)

        # Build shapes
        shapes = self._build_shapes(fills, strokes, paths)

        layer = {
            "ddd": 0,
            "ind": index + 1,
            "ty": ty,
            "nm": nm,
            "sr": 1,
            "ks": ks,
            "ao": 0,
            "ip": 0,
            "op": self.dur,
            "st": 0,
            "bm": 0
        }

        if shapes:
            layer["shapes"] = shapes

        return layer

    def _parse_keyframes(self, data: str) -> tuple:
        """Parse keyframe string. Supports:
        - Animated: '0.0→[0.5,0.5] 0.5→[0.8,0.3] ease=smooth'
        - Static compact: '[0.5,0.5]' or '720'
        Returns (keyframes_list, ease_type)"""
        if not data:
            return [], 'none'

        ease_match = re.search(r'ease=(\w+)', data)
        ease = ease_match.group(1) if ease_match else 'none'

        # Remove ease part
        kf_str = re.sub(r'\s*ease=\w+', '', data).strip()

        keyframes = []
        # Match animated patterns like 0.0→[0.5,0.5] or 0.0→720.0
        pattern = r'([\d.]+)\u2192(\[[\d.,\s-]+\]|[\d.\-]+)'
        for match in re.finditer(pattern, kf_str):
            t = float(match.group(1))
            val_str = match.group(2)
            if val_str.startswith('['):
                nums = re.findall(r'[\d.\-]+', val_str)
                val = [float(n) for n in nums]
            else:
                val = float(val_str)
            keyframes.append((t, val))

        # If no arrow-style keyframes found, try static compact format
        if not keyframes:
            arr_match = re.match(r'\[([\d.,\s-]+)\]', kf_str)
            if arr_match:
                nums = re.findall(r'[\d.\-]+', arr_match.group(1))
                keyframes.append((0.0, [float(n) for n in nums]))
            else:
                try:
                    keyframes.append((0.0, float(kf_str)))
                except ValueError:
                    pass

        return keyframes, ease

    def _ease_to_bezier(self, ease: str) -> tuple:
        """Convert ease name to Lottie bezier in/out curves."""
        if ease == 'smooth':
            return ({"x": [0.667], "y": [1]}, {"x": [0.333], "y": [0]})
        elif ease == 'bounce':
            return ({"x": [0.667], "y": [1.5]}, {"x": [0.333], "y": [0]})
        elif ease == 'fade':
            return ({"x": [0.667], "y": [1]}, {"x": [0.333], "y": [0]})
        else:  # linear
            return ({"x": [1], "y": [1]}, {"x": [0], "y": [0]})

    def _build_pos_prop(self, data: str) -> dict:
        """Build Lottie position property from AnimTOON pos data."""
        if not data:
            return {"a": 0, "k": [self.w / 2, self.h / 2, 0]}

        keyframes, ease = self._parse_keyframes(data)
        if not keyframes:
            return {"a": 0, "k": [self.w / 2, self.h / 2, 0]}

        if len(keyframes) == 1 or ease == 'none':
            t, val = keyframes[0]
            if isinstance(val, list):
                x = val[0] * self.w
                y = val[1] * self.h if len(val) > 1 else val[0] * self.h
                return {"a": 0, "k": [round(x, 3), round(y, 3), 0]}
            return {"a": 0, "k": [self.w / 2, self.h / 2, 0]}

        # Animated
        i_curve, o_curve = self._ease_to_bezier(ease)
        lottie_kfs = []
        for idx, (t, val) in enumerate(keyframes):
            frame = round(t * self.dur)
            if isinstance(val, list):
                x = round(val[0] * self.w, 3)
                y = round(val[1] * self.h, 3) if len(val) > 1 else round(val[0] * self.h, 3)
                s = [x, y, 0]
            else:
                s = [round(val * self.w, 3), round(val * self.h, 3), 0]

            kf = {"t": frame, "s": s}
            if idx < len(keyframes) - 1:
                kf["i"] = i_curve
                kf["o"] = o_curve
                # Spatial tangents
                kf["to"] = [0, 0, 0]
                kf["ti"] = [0, 0, 0]
            lottie_kfs.append(kf)

        return {"a": 1, "k": lottie_kfs}

    def _build_anchor_from_pos(self, data: str) -> dict:
        """Build anchor point at [0,0] so shapes scale/rotate around their center.
        Position handles placement on canvas, anchor stays at shape origin."""
        return {"a": 0, "k": [0, 0, 0]}

    def _build_scalar_prop(self, data: str, default) -> dict:
        """Build Lottie scalar property (rotation/opacity)."""
        if not data:
            return {"a": 0, "k": default}

        keyframes, ease = self._parse_keyframes(data)
        if not keyframes:
            return {"a": 0, "k": default}

        if len(keyframes) == 1 or ease == 'none':
            _, val = keyframes[0]
            if isinstance(val, list):
                val = val[0] if val else default
            return {"a": 0, "k": round(val, 1) if isinstance(val, float) else val}

        # Animated
        i_curve, o_curve = self._ease_to_bezier(ease)
        lottie_kfs = []
        for idx, (t, val) in enumerate(keyframes):
            frame = round(t * self.dur)
            if isinstance(val, list):
                val = val[0] if val else default
            s_val = [round(val, 1) if isinstance(val, float) else val]
            kf = {"t": frame, "s": s_val}
            if idx < len(keyframes) - 1:
                kf["i"] = i_curve
                kf["o"] = o_curve
            lottie_kfs.append(kf)

        return {"a": 1, "k": lottie_kfs}

    def _build_scale_prop(self, data: str) -> dict:
        """Build Lottie scale property."""
        if not data:
            return {"a": 0, "k": [100, 100, 100]}

        keyframes, ease = self._parse_keyframes(data)
        if not keyframes:
            return {"a": 0, "k": [100, 100, 100]}

        if len(keyframes) == 1 or ease == 'none':
            _, val = keyframes[0]
            if isinstance(val, list) and len(val) >= 2:
                return {"a": 0, "k": [val[0], val[1], 100]}
            v = val if not isinstance(val, list) else 100
            return {"a": 0, "k": [v, v, 100]}

        # Animated
        i_curve, o_curve = self._ease_to_bezier(ease)
        lottie_kfs = []
        for idx, (t, val) in enumerate(keyframes):
            frame = round(t * self.dur)
            if isinstance(val, list) and len(val) >= 2:
                s = [val[0], val[1], 100]
            else:
                v = val if not isinstance(val, list) else 100
                s = [v, v, 100]
            kf = {"t": frame, "s": s}
            if idx < len(keyframes) - 1:
                kf["i"] = {"x": [0.667, 0.667, 0.667], "y": [1, 1, 1]}
                kf["o"] = {"x": [0.333, 0.333, 0.333], "y": [0, 0, 0]}
            lottie_kfs.append(kf)

        return {"a": 1, "k": lottie_kfs}

    def _hex_to_lottie_color(self, hex_str: str) -> list:
        """Convert #RRGGBB to Lottie [r, g, b, 1] (0-1 range)."""
        hex_str = hex_str.strip('#')
        if len(hex_str) != 6:
            return [0, 0, 0, 1]
        r = int(hex_str[0:2], 16) / 255.0
        g = int(hex_str[2:4], 16) / 255.0
        b = int(hex_str[4:6], 16) / 255.0
        return [round(r, 6), round(g, 6), round(b, 6), 1]

    def _build_shapes(self, fills: list, strokes: list, paths: list) -> list:
        """Build Lottie shapes array from AnimTOON fill/stroke/path data."""
        items = []

        # Build path shapes
        for p in paths:
            items.append(self._build_path(p))

        # Build fills (place AFTER paths - important for rendering)
        for f in fills:
            color = self._hex_to_lottie_color(f)
            items.append({
                "ty": "fl",
                "c": {"a": 0, "k": color},
                "o": {"a": 0, "k": 100},
                "r": 1,
                "bm": 0,
                "nm": "Fill 1",
                "hd": False
            })

        # Build strokes
        for s in strokes:
            parts = s.split()
            color_hex = parts[0] if parts else '#000000'
            w_val = 1.0
            for part in parts[1:]:
                if part.startswith('w='):
                    try:
                        w_val = float(part[2:]) * self.w
                    except ValueError:
                        w_val = 1.0
            color = self._hex_to_lottie_color(color_hex)
            items.append({
                "ty": "st",
                "c": {"a": 0, "k": color},
                "o": {"a": 0, "k": 100},
                "w": {"a": 0, "k": round(w_val, 2)},
                "lc": 1,
                "lj": 1,
                "ml": 10,
                "bm": 0,
                "nm": "Stroke 1",
                "hd": False
            })

        if not items:
            return []

        # Wrap in a group with transform
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

        return [{"ty": "gr", "it": items, "nm": "Group 1", "bm": 0, "hd": False}]

    def _build_path(self, path_str: str) -> dict:
        """Build a Lottie path shape from AnimTOON path description."""
        parts = path_str.strip().split()
        path_type = parts[0] if parts else 'sh'

        if path_type == 'ellipse':
            w = 0.1
            h = 0.1
            for p in parts[1:]:
                if p.startswith('w='):
                    try:
                        w = float(p[2:])
                    except ValueError:
                        pass
                elif p.startswith('h='):
                    try:
                        h = float(p[2:])
                    except ValueError:
                        pass
            return {
                "ty": "el",
                "s": {"a": 0, "k": [round(w * self.w, 2), round(h * self.h, 2)]},
                "p": {"a": 0, "k": [0, 0]},
                "nm": "Ellipse 1",
                "hd": False
            }
        elif path_type == 'rect':
            w = 0.1
            h = 0.1
            r = 0
            for p in parts[1:]:
                if p.startswith('w='):
                    try:
                        w = float(p[2:])
                    except ValueError:
                        pass
                elif p.startswith('h='):
                    try:
                        h = float(p[2:])
                    except ValueError:
                        pass
                elif p.startswith('r='):
                    try:
                        r = float(p[2:])
                    except ValueError:
                        pass
            return {
                "ty": "rc",
                "s": {"a": 0, "k": [round(w * self.w, 2), round(h * self.h, 2)]},
                "p": {"a": 0, "k": [0, 0]},
                "r": {"a": 0, "k": round(r * self.w, 2)},
                "nm": "Rect 1",
                "hd": False
            }
        else:
            # Default: ellipse shape (~20% of canvas, renders as clean circle)
            size = round(self.w * 0.2)  # 20% of canvas
            return {
                "ty": "el",
                "s": {"a": 0, "k": [size, size]},
                "p": {"a": 0, "k": [0, 0]},
                "nm": "Ellipse 1",
                "hd": False
            }


# ─── VALIDATION & FIX ───────────────────────────────────────────────

def validate_and_fix(lottie_json: dict, canvas_w: int = 512, canvas_h: int = 512) -> tuple:
    """Validate recovered Lottie JSON and fix common issues from model generation.

    Fixes:
      1. Layer with shapes but no fill → add #808080 fill
      2. Opacity > 100 → clamp to 100
      3. Opacity < 0 → clamp to 0
      4. Position x/y outside canvas range → clamp to 0–canvas
      5. Layer with empty shapes array → add default ellipse

    Args:
        lottie_json: Lottie JSON dict (output of AnimTOONToLottie.convert)
        canvas_w: Canvas width (default 512)
        canvas_h: Canvas height (default 512)

    Returns:
        (fixed_lottie_json, fixes_applied) where fixes_applied is a list of strings
    """
    fixes = []

    for layer in lottie_json.get('layers', []):
        nm = layer.get('nm', 'unnamed')

        # --- Fix 5: empty shapes array → add default ellipse ---
        if 'shapes' in layer and isinstance(layer['shapes'], list) and len(layer['shapes']) == 0:
            layer['shapes'] = [{
                "ty": "gr",
                "it": [
                    {
                        "ty": "el",
                        "s": {"a": 0, "k": [50, 50]},
                        "p": {"a": 0, "k": [0, 0]},
                        "nm": "Ellipse 1",
                        "hd": False,
                    },
                    {
                        "ty": "fl",
                        "c": {"a": 0, "k": [0.502, 0.502, 0.502, 1]},
                        "o": {"a": 0, "k": 100},
                        "r": 1,
                        "bm": 0,
                        "nm": "Fill 1",
                        "hd": False,
                    },
                    {
                        "ty": "tr",
                        "p": {"a": 0, "k": [0, 0]},
                        "a": {"a": 0, "k": [0, 0]},
                        "s": {"a": 0, "k": [100, 100]},
                        "r": {"a": 0, "k": 0},
                        "o": {"a": 0, "k": 100},
                        "sk": {"a": 0, "k": 0},
                        "sa": {"a": 0, "k": 0},
                        "nm": "Transform",
                    },
                ],
                "nm": "Group 1",
                "bm": 0,
                "hd": False,
            }]
            fixes.append(f"[{nm}] empty shapes → added default ellipse")

        # --- Fix 1: shapes with paths but no fill → add #808080 ---
        if 'shapes' in layer and isinstance(layer['shapes'], list):
            for group in layer['shapes']:
                if group.get('ty') != 'gr':
                    continue
                items = group.get('it', [])
                has_path = any(it.get('ty') in ('sh', 'el', 'rc') for it in items)
                has_fill = any(it.get('ty') == 'fl' for it in items)
                has_stroke = any(it.get('ty') == 'st' for it in items)
                if has_path and not has_fill and not has_stroke:
                    # Insert fill before the transform (tr) element
                    fill = {
                        "ty": "fl",
                        "c": {"a": 0, "k": [0.502, 0.502, 0.502, 1]},
                        "o": {"a": 0, "k": 100},
                        "r": 1,
                        "bm": 0,
                        "nm": "Fill 1",
                        "hd": False,
                    }
                    tr_idx = next((i for i, it in enumerate(items) if it.get('ty') == 'tr'), len(items))
                    items.insert(tr_idx, fill)
                    fixes.append(f"[{nm}] no fill/stroke → added #808080 fill")

        # --- Fix 2 & 3: clamp opacity ---
        ks = layer.get('ks', {})
        o_prop = ks.get('o', {})
        if o_prop:
            _clamp_opacity(o_prop, nm, fixes)

        # --- Fix 4: clamp position to canvas ---
        p_prop = ks.get('p', {})
        if p_prop:
            _clamp_position(p_prop, canvas_w, canvas_h, nm, fixes)

    return lottie_json, fixes


def _clamp_opacity(prop: dict, layer_name: str, fixes: list):
    """Clamp opacity values to 0–100 range."""
    if prop.get('a', 0) == 1:
        # Animated keyframes
        for kf in prop.get('k', []):
            if not isinstance(kf, dict):
                continue
            s = kf.get('s')
            if isinstance(s, list):
                for i in range(len(s)):
                    if isinstance(s[i], (int, float)):
                        if s[i] > 100:
                            fixes.append(f"[{layer_name}] opacity {s[i]} → clamped to 100")
                            s[i] = 100
                        elif s[i] < 0:
                            fixes.append(f"[{layer_name}] opacity {s[i]} → clamped to 0")
                            s[i] = 0
    else:
        # Static value
        k = prop.get('k')
        if isinstance(k, (int, float)):
            if k > 100:
                fixes.append(f"[{layer_name}] opacity {k} → clamped to 100")
                prop['k'] = 100
            elif k < 0:
                fixes.append(f"[{layer_name}] opacity {k} → clamped to 0")
                prop['k'] = 0


def _clamp_position(prop: dict, w: int, h: int, layer_name: str, fixes: list):
    """Clamp position values to 0–canvas range."""
    if prop.get('a', 0) == 1:
        # Animated keyframes
        for kf in prop.get('k', []):
            if not isinstance(kf, dict):
                continue
            s = kf.get('s')
            if isinstance(s, list) and len(s) >= 2:
                clamped = False
                if s[0] < 0:
                    s[0] = 0
                    clamped = True
                elif s[0] > w:
                    s[0] = w
                    clamped = True
                if s[1] < 0:
                    s[1] = 0
                    clamped = True
                elif s[1] > h:
                    s[1] = h
                    clamped = True
                if clamped:
                    fixes.append(f"[{layer_name}] position out of canvas → clamped")
    else:
        # Static value
        k = prop.get('k')
        if isinstance(k, list) and len(k) >= 2:
            clamped = False
            if k[0] < 0:
                k[0] = 0
                clamped = True
            elif k[0] > w:
                k[0] = w
                clamped = True
            if k[1] < 0:
                k[1] = 0
                clamped = True
            elif k[1] > h:
                k[1] = h
                clamped = True
            if clamped:
                fixes.append(f"[{layer_name}] position out of canvas → clamped")


# ─── TOKEN COUNTING ─────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """Rough token count estimate (4 chars per token average)."""
    return len(text) // 4


# ─── MULTI-ANIMATION DOTLOTTIE ─────────────────────────────────────

def dotlottie_to_animtoon_full(dotlottie_path: str) -> str:
    """Convert a dotLottie (possibly multi-animation) to unified AnimTOON text.

    Reads manifest.json for animation list and state machines,
    converts each animation JSON to AnimTOON, and combines into
    the multi-animation dotlottie format:

        dotlottie version=1.0

        anim id=idle fr=30 dur=60 loop=true
          layer ball shape
            fill #0000FF
            ...

        states
          idle onClick→bounce
          bounce onComplete→idle
    """
    with zipfile.ZipFile(dotlottie_path, 'r') as z:
        manifest = json.loads(z.read('manifest.json').decode('utf-8'))

        # Collect all animation JSONs
        animations = []
        for anim_entry in manifest.get('animations', []):
            anim_id = anim_entry.get('id', 'default')
            loop = anim_entry.get('loop', False)
            # Find the JSON file — try standard paths
            anim_file = None
            for candidate in [
                f"animations/{anim_id}.json",
                f"{anim_id}.json",
            ]:
                if candidate in z.namelist():
                    anim_file = candidate
                    break
            # Fallback: grab any JSON that isn't manifest
            if anim_file is None:
                json_files = [f for f in z.namelist()
                              if f.endswith('.json') and f != 'manifest.json']
                if json_files:
                    anim_file = json_files[0]
            if anim_file is None:
                continue

            lottie_json = json.loads(z.read(anim_file).decode('utf-8'))
            animations.append({
                'id': anim_id,
                'loop': loop,
                'lottie': lottie_json,
                'file': anim_file,
            })

        # Extract state machines from manifest
        states = []
        for sm in manifest.get('states', []):
            # dotLottie state machine format varies; extract transitions
            state_id = sm.get('id', '')
            # Some manifests embed descriptor JSON
            descriptor = sm.get('descriptor', {})
            if isinstance(descriptor, str):
                try:
                    descriptor = json.loads(descriptor)
                except (json.JSONDecodeError, TypeError):
                    descriptor = {}
            for st_name, st_def in descriptor.get('states', {}).items():
                transitions = st_def.get('transitions', [])
                for tr in transitions:
                    trigger = tr.get('type', 'onClick')
                    target = tr.get('to', '')
                    if target:
                        states.append(f"  {st_name} {trigger}→{target}")

    # Build unified AnimTOON text
    converter = LottieToAnimTOON()
    lines = ["dotlottie version=1.0", ""]

    for anim in animations:
        lottie = anim['lottie']
        fr = int(lottie.get('fr', 60))
        ip = lottie.get('ip', 0)
        op = lottie.get('op', 360)
        dur = int(op - ip)
        loop_str = "true" if anim['loop'] else "false"

        lines.append(f"anim id={anim['id']} fr={fr} dur={dur} loop={loop_str}")

        # Convert this animation's layers to AnimTOON lines
        single_text = converter.convert(lottie)
        # Strip the single-anim "anim fr=X dur=Y" header, keep only layer blocks
        for line in single_text.split('\n'):
            if line.strip().startswith('anim '):
                continue  # skip the old single-anim header
            if line.strip() == '':
                lines.append('')
            else:
                # Indent layer content by 2 spaces under the anim block
                lines.append(f"  {line}")

    # Add state machine block if present
    if states:
        # Remove trailing blank lines before states
        while lines and lines[-1].strip() == '':
            lines.pop()
        lines.append('')
        lines.append('states')
        lines.extend(states)

    return '\n'.join(lines).strip() + '\n'


def animtoon_to_dotlottie_full(animtoon_text: str, output_path: str):
    """Convert unified multi-animation AnimTOON text back to dotLottie ZIP.

    Parses the dotlottie header, splits anim blocks, converts each to
    Lottie JSON, rebuilds manifest with state machines, and packs as ZIP.
    """
    lines = animtoon_text.strip().split('\n')

    # Parse dotlottie header
    version = "1.0"
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('dotlottie'):
            ver_match = re.search(r'version=([\d.]+)', stripped)
            if ver_match:
                version = ver_match.group(1)
            break

    # Split into anim blocks and states block
    anim_blocks = []  # list of (header_str, [layer_lines])
    state_lines = []
    current_header = None
    current_lines = []
    in_states = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith('dotlottie'):
            continue

        if stripped.startswith('states'):
            # Save current anim block
            if current_header is not None:
                anim_blocks.append((current_header, current_lines))
                current_header = None
                current_lines = []
            in_states = True
            continue

        if in_states:
            if stripped:
                state_lines.append(stripped)
            continue

        # Check for anim header (at start of line, not indented)
        if re.match(r'^anim\s', stripped) and not line.startswith('  '):
            # Save previous block
            if current_header is not None:
                anim_blocks.append((current_header, current_lines))
            current_header = stripped
            current_lines = []
        elif current_header is not None:
            # Content line — remove the 2-space indent from dotlottie format
            if line.startswith('  '):
                current_lines.append(line[2:])
            else:
                current_lines.append(line)

    # Don't forget last block
    if current_header is not None:
        anim_blocks.append((current_header, current_lines))

    # Convert each anim block
    reverse_converter = AnimTOONToLottie()
    manifest_anims = []
    anim_files = {}  # filename -> lottie_json

    for header_str, layer_lines in anim_blocks:
        # Parse header: anim id=X fr=Y dur=Z loop=true/false
        id_match = re.search(r'id=(\S+)', header_str)
        fr_match = re.search(r'fr=(\d+)', header_str)
        dur_match = re.search(r'dur=(\d+)', header_str)
        loop_match = re.search(r'loop=(\w+)', header_str)

        anim_id = id_match.group(1) if id_match else 'default'
        fr = int(fr_match.group(1)) if fr_match else 60
        dur = int(dur_match.group(1)) if dur_match else 360
        loop = (loop_match.group(1).lower() == 'true') if loop_match else False

        # Rebuild single-anim AnimTOON text with header for the converter
        single_text = f"anim fr={fr} dur={dur}\n" + '\n'.join(layer_lines)
        lottie_json = reverse_converter.convert(single_text)

        # Validate and fix
        lottie_json, _ = validate_and_fix(lottie_json)

        filename = f"animations/{anim_id}.json"
        anim_files[filename] = lottie_json
        manifest_anims.append({
            "id": anim_id,
            "loop": loop,
        })

    # Parse state machines
    state_machine = None
    if state_lines:
        states_dict = {}
        for sl in state_lines:
            # Format: "idle onClick→bounce"
            parts = sl.split()
            if len(parts) >= 2:
                state_name = parts[0]
                transitions = []
                for transition_str in parts[1:]:
                    arrow_match = re.match(r'(\w+)→(\w+)', transition_str)
                    if arrow_match:
                        trigger = arrow_match.group(1)
                        target = arrow_match.group(2)
                        transitions.append({
                            "type": trigger,
                            "to": target,
                        })
                if transitions:
                    states_dict[state_name] = {"transitions": transitions}

        if states_dict:
            state_machine = {
                "id": "default",
                "descriptor": {
                    "initial": list(states_dict.keys())[0],
                    "states": states_dict,
                },
            }

    # Build manifest
    manifest = {
        "version": version,
        "generator": "AnimTOON",
        "animations": manifest_anims,
    }
    if state_machine:
        manifest["states"] = [state_machine]

    # Pack as dotLottie ZIP
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json', json.dumps(manifest, indent=2))
        for filename, lottie_json in anim_files.items():
            z.writestr(filename, json.dumps(lottie_json))


# ─── PIPELINE ────────────────────────────────────────────────────────

def run_pipeline(input_path: str, output_path: str):
    """
    Full pipeline:
    dotLottie → Lottie JSON → AnimTOON → Lottie JSON → dotLottie
    """
    print("=== AnimTOON Pipeline ===")
    input_size = os.path.getsize(input_path)
    print(f"Input: {os.path.basename(input_path)} ({input_size // 1024} KB)")
    print()

    # Step 1: Unzip dotLottie
    print("Step 1: Unzipped dotLottie")
    lottie_json, manifest, anim_name = dotlottie_to_lottie(input_path)
    w = lottie_json.get('w', 512)
    h = lottie_json.get('h', 512)
    fr = lottie_json.get('fr', 60)
    ip = lottie_json.get('ip', 0)
    op = lottie_json.get('op', 360)
    dur_sec = (op - ip) / fr if fr > 0 else 0
    num_layers = len(lottie_json.get('layers', []))
    print(f"  Found animation: {anim_name}")
    print(f"  Canvas: {w}x{h}, FPS: {fr}, Duration: {dur_sec:.1f}s, Layers: {num_layers}")
    print()

    # Step 2: Convert to AnimTOON (multi-animation format)
    print("Step 2: Converted to AnimTOON (dotlottie format)")
    animtoon_text = dotlottie_to_animtoon_full(input_path)
    animtoon_chars = len(animtoon_text)
    animtoon_tokens = count_tokens(animtoon_text)
    original_json_str = json.dumps(lottie_json)
    original_tokens = count_tokens(original_json_str)
    reduction = (1 - animtoon_tokens / original_tokens) * 100 if original_tokens > 0 else 0
    print(f"  AnimTOON text: {animtoon_chars:,} chars")
    print(f"  Estimated tokens: {animtoon_tokens:,}")
    print(f"  Original JSON tokens: {original_tokens:,}")
    print(f"  Token reduction: {reduction:.1f}%")
    print()

    # Print the AnimTOON text (first 2000 chars)
    print("--- AnimTOON Output (preview) ---")
    preview = animtoon_text[:2000]
    if len(animtoon_text) > 2000:
        preview += f"\n... ({len(animtoon_text) - 2000} more chars)"
    print(preview)
    print("--- End Preview ---")
    print()

    # Step 3: Convert back to dotLottie
    print("Step 3: Converted back to dotLottie")
    animtoon_to_dotlottie_full(animtoon_text, output_path)
    # Read back to verify
    recovered_json, _, _ = dotlottie_to_lottie(output_path)
    recovered_layers = len(recovered_json.get('layers', []))
    is_valid = True
    try:
        json.dumps(recovered_json)
    except Exception:
        is_valid = False
    print(f"  Layers recovered: {recovered_layers}/{num_layers}")
    print(f"  Valid JSON: {'YES' if is_valid else 'NO'}")
    output_size = os.path.getsize(output_path)
    print(f"  Output: {os.path.basename(output_path)} ({output_size // 1024} KB)")
    print()

    # Step 4: Verification
    print("Step 4: Verification")
    print(f"  Original layers: {num_layers}")
    print(f"  Recovered layers: {recovered_layers}")
    success = recovered_layers == num_layers and is_valid
    print(f"  Round trip: {'SUCCESS' if success else 'PARTIAL (layer count mismatch)'}")

    # Save AnimTOON text for inspection
    txt_path = output_path.replace('.lottie', '.animtoon.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(animtoon_text)
    print(f"\n  AnimTOON text saved to: {txt_path}")

    # Token comparison summary
    print()
    print("=== Token Comparison ===")
    print(f"  Original Lottie JSON:  {original_tokens:,} tokens")
    print(f"  AnimTOON dotlottie:    {animtoon_tokens:,} tokens")
    print(f"  Reduction:             {reduction:.1f}%")

    return animtoon_text, recovered_json


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace('.lottie', '_animtoon.lottie')
    else:
        # Default paths
        base = os.path.dirname(os.path.abspath(__file__))
        input_path = os.path.join(base, "Error 404.lottie")
        output_path = os.path.join(base, "Error_404_animtoon.lottie")

    run_pipeline(input_path, output_path)
