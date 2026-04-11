"""Post-process scraped Lottie JSONs into AnimTOON v4 training pairs.

Input:  data/private_scraped/iconscout/{category}/{id}.json
Output: data/private_scraped/animtoon_v4_scraped.jsonl  (gitignored)

Each output record:
  {
    "prompt": "<hierarchical description>",
    "output": "<AnimTOON v4 text>",
    "meta": { "id", "category", "source", "v", "layers", ... }
  }

Prompt generation is hierarchical:
  Level 1 (character): "A {category} animation with {N} layers"
  Level 2 (structure): detected body parts / layer types
  Level 3 (motion):    summarized keyframe patterns
  Level 4 (timing):    duration, fps, easing

Also strips metadata fields (nm, source IDs) so nothing traces back to
IconScout in the training data.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Add src to path so we can reuse the converter
sys.path.insert(0, str(Path(__file__).parent))
from toon_animator import LottieToAnimTOON, count_tokens

SCRAPED_ROOT = Path("data/private_scraped/iconscout")
OUTPUT_FILE = Path("data/private_scraped/animtoon_v4_scraped.jsonl")

# Heuristic character-part detection from layer names
BODY_PART_HINTS = {
    "head": ["head", "face", "cabeza"],
    "body": ["body", "torso", "chest", "cuerpo"],
    "arm": ["arm", "hand", "brazo", "mano"],
    "leg": ["leg", "foot", "pierna", "pie"],
    "eye": ["eye", "pupil", "ojo"],
    "mouth": ["mouth", "lip", "boca"],
    "hair": ["hair", "cabello"],
    "tail": ["tail", "cola"],
    "wing": ["wing", "ala"],
    "ear": ["ear", "oreja"],
}

# Motion pattern detection from keyframes
MOTION_HINTS = {
    "bounce": ["bounce", "hop", "jump"],
    "rotate": ["rotate", "spin", "turn"],
    "wave":  ["wave", "hand", "greet"],
    "walk":  ["walk", "step"],
    "idle":  ["idle", "breath", "stand"],
    "scale": ["scale", "grow", "shrink"],
    "fade":  ["fade", "appear", "disappear"],
}


def sanitize_layer_name(nm):
    """Lowercase, keep alnum + underscore, no spaces."""
    if not nm:
        return "layer"
    s = re.sub(r"[^\w]+", "_", str(nm).strip().lower())
    return s.strip("_") or "layer"


def detect_body_parts(layers):
    """Count detected body-part layer types."""
    counts = Counter()
    for l in layers:
        nm = str(l.get("nm", "")).lower()
        for part, hints in BODY_PART_HINTS.items():
            if any(h in nm for h in hints):
                counts[part] += 1
                break
    return counts


def detect_motion_type(lottie):
    """Infer motion type from keyframe patterns + layer count + duration."""
    layers = lottie.get("layers", [])
    fr = lottie.get("fr", 30)
    op = lottie.get("op", 0)
    ip = lottie.get("ip", 0)
    duration_s = (op - ip) / max(fr, 1)

    # Check which properties are animated
    has_rot = has_pos = has_scale = has_opacity = False
    for l in layers:
        ks = l.get("ks") or {}
        if isinstance(ks.get("r", {}), dict) and ks["r"].get("a") == 1:
            has_rot = True
        if isinstance(ks.get("p", {}), dict) and ks["p"].get("a") == 1:
            has_pos = True
        if isinstance(ks.get("s", {}), dict) and ks["s"].get("a") == 1:
            has_scale = True
        if isinstance(ks.get("o", {}), dict) and ks["o"].get("a") == 1:
            has_opacity = True

    motions = []
    if has_rot:
        motions.append("rotation")
    if has_pos:
        motions.append("translation")
    if has_scale:
        motions.append("scaling")
    if has_opacity:
        motions.append("opacity")

    # Heuristic named motion
    if has_rot and has_pos and duration_s <= 3:
        named = "expressive motion"
    elif has_scale and not has_pos:
        named = "pulse"
    elif has_pos and not has_rot:
        named = "translate"
    elif has_rot and not has_pos:
        named = "rotation cycle"
    elif has_opacity and len(motions) == 1:
        named = "fade"
    elif not motions:
        named = "static"
    else:
        named = "mixed motion"

    return {
        "named": named,
        "properties": motions,
        "duration_s": round(duration_s, 1),
        "fps": fr,
        "has_rot": has_rot,
        "has_pos": has_pos,
        "has_scale": has_scale,
    }


def build_prompt(category, lottie, parts, motion):
    """Hierarchical prompt — character structure + motion + style."""
    layers = lottie.get("layers", [])
    n_layers = len(layers)
    w = lottie.get("w", 0)
    h = lottie.get("h", 0)

    # Level 1: subject
    if parts:
        part_desc = ", ".join(f"{v} {k}" for k, v in parts.most_common(6))
        subject = f"A {category} character with {n_layers} layers ({part_desc})"
    else:
        subject = f"A {category} animation with {n_layers} layers"

    # Level 2: style cue from canvas size
    if w == h and w <= 256:
        style = "simple icon-sized"
    elif w == h and w <= 1024:
        style = "medium-complexity"
    else:
        style = "full-canvas"

    # Level 3: motion
    motion_desc = motion["named"]
    if motion["properties"]:
        motion_desc += " (" + ", ".join(motion["properties"]) + ")"

    # Level 4: timing
    timing = f"{motion['duration_s']}s at {motion['fps']}fps"

    return (
        f"{subject}. Style: {style}. "
        f"Motion: {motion_desc}. Timing: {timing}."
    )


def strip_metadata(lottie):
    """Remove identifying metadata from a Lottie before conversion."""
    # Rename all layer nm to generic layer_N
    for i, l in enumerate(lottie.get("layers", [])):
        if "nm" in l:
            l["nm"] = f"layer_{i}"
    # Strip top-level metadata
    for k in ["nm", "meta", "markers", "metadata"]:
        lottie.pop(k, None)
    return lottie


def process_one(json_path, category, converter):
    """Convert one scraped Lottie into a training record."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8-sig"))
    except Exception as e:
        return None, f"parse: {e}"

    if not isinstance(data, dict) or "layers" not in data or "ip" not in data:
        return None, "not lottie"

    # Detect BEFORE stripping (we use original names to infer parts)
    parts = detect_body_parts(data.get("layers", []))
    motion = detect_motion_type(data)

    # Build hierarchical prompt
    prompt = build_prompt(category, data, parts, motion)

    # Strip identifying metadata
    data = strip_metadata(data)

    # Convert to AnimTOON v4
    try:
        animtoon = converter.convert(data)
    except Exception as e:
        return None, f"convert: {e}"

    record = {
        "prompt": prompt,
        "output": animtoon,
        "meta": {
            "category": category,
            "v": data.get("v"),
            "w": data.get("w"),
            "h": data.get("h"),
            "fr": data.get("fr"),
            "layer_count": len(data.get("layers", [])),
            "token_count": count_tokens(animtoon),
            "parts": dict(parts),
            "motion": motion["named"],
        }
    }
    return record, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", help="only process this category")
    parser.add_argument("--limit", type=int, help="stop after N files per category")
    parser.add_argument("--min-layers", type=int, default=2,
                        help="skip files with fewer layers than this")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="skip outputs longer than this many tokens")
    args = parser.parse_args()

    if not SCRAPED_ROOT.exists():
        print(f"No scraped data at {SCRAPED_ROOT}")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    converter = LottieToAnimTOON()

    stats = Counter()
    errors = Counter()
    written = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        categories = [args.category] if args.category else sorted(
            [d.name for d in SCRAPED_ROOT.iterdir() if d.is_dir()]
        )
        for cat in categories:
            cat_dir = SCRAPED_ROOT / cat
            if not cat_dir.exists():
                continue
            files = [f for f in sorted(cat_dir.glob("*.json"))
                     if not f.name.endswith(".meta.json")]
            if args.limit:
                files = files[:args.limit]

            print(f"\n[{cat}] processing {len(files)} files...")
            for f in files:
                record, err = process_one(f, cat, converter)
                stats[cat] += 1
                if err:
                    errors[err[:40]] += 1
                    continue

                # Quality filters
                if record["meta"]["layer_count"] < args.min_layers:
                    errors["too_few_layers"] += 1
                    continue
                if record["meta"]["token_count"] > args.max_tokens:
                    errors["too_many_tokens"] += 1
                    continue

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

                if written % 200 == 0:
                    print(f"  written {written}")

    print(f"\n{'='*50}")
    print(f"Processed: {sum(stats.values())}")
    print(f"Written:   {written}")
    print(f"Output:    {OUTPUT_FILE}")
    print(f"\nSkipped reasons:")
    for reason, n in errors.most_common(10):
        print(f"  {n:5d}  {reason}")


if __name__ == "__main__":
    main()
