"""Build the training JSONL for v3.1 / v4.1 training.

Input:  data/private_scraped/iconscout/{category}/{id}.json   (21,554 files)
Output: data/iconscout_training_v4.jsonl                       (training pairs)

For each valid Lottie file:
  1. Convert to AnimTOON v4 (integer format, parent= preserved)
  2. Drop if > 1024 tokens (won't fit model context)
  3. Extract facts (layer count, parts, motion type, duration, animated props)
  4. Generate 4 prompts describing the SAME animation at different detail levels:
       terse_natural, terse_structural, detailed_natural, detailed_structural
  5. Emit 4 training records per file (one per prompt) — same output, different prompts

The model learns that different phrasings of the same animation map to the same
output. This teaches prompt understanding, not keyword matching.

No layer-count filter, no duration filter — keep everything that fits in 1024 tokens.
PRIVATE: data/ is gitignored, this JSONL never leaves your machine for publication.
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from toon_animator import LottieToAnimTOON, count_tokens

SCRAPED_ROOT = Path("data/private_scraped/iconscout")
OUTPUT_FILE = Path("data/iconscout_training_v4.jsonl")

MAX_OUTPUT_TOKENS = 1024  # HARD: outputs longer than this are dropped


# ---------- Fact extraction from Lottie ----------

BODY_PART_HINTS = {
    "head": ["head", "face", "hair", "cabeza", "cabello"],
    "eye": ["eye", "pupil", "iris", "ojo"],
    "mouth": ["mouth", "lip", "teeth", "tongue", "boca"],
    "ear": ["ear", "oreja"],
    "torso": ["torso", "body", "chest", "shirt", "cuerpo", "dress"],
    "arm": ["arm", "shoulder", "elbow", "brazo"],
    "hand": ["hand", "finger", "palm", "mano"],
    "leg": ["leg", "thigh", "knee", "pierna"],
    "foot": ["foot", "shoe", "pie", "boot"],
    "tail": ["tail", "cola"],
    "wing": ["wing", "ala"],
    "horn": ["horn", "antler"],
    "hat": ["hat", "cap", "crown"],
}

MOTION_VERBS = {
    "walk":   ["walk", "step", "stride"],
    "run":    ["run", "sprint", "dash"],
    "wave":   ["wave", "greet", "hello"],
    "bounce": ["bounce", "hop", "jump"],
    "dance":  ["dance"],
    "fly":    ["fly", "soar"],
    "swim":   ["swim"],
    "blink":  ["blink"],
    "breathe":["breath", "idle"],
    "spin":   ["spin", "rotate", "turn"],
    "fade":   ["fade", "appear", "disappear"],
    "pulse":  ["pulse", "beat", "throb"],
    "shake":  ["shake", "wobble"],
}


def detect_body_parts(layers):
    """Count each body-part type present in layer names."""
    counts = Counter()
    for l in layers:
        nm = str(l.get("nm", "")).lower()
        if not nm:
            continue
        for part, hints in BODY_PART_HINTS.items():
            if any(h in nm for h in hints):
                counts[part] += 1
                break
    return counts


def detect_animated_props(layers):
    """Return set of animated property names: rot/pos/scale/opacity."""
    props = set()
    for l in layers:
        ks = l.get("ks") or {}
        if isinstance(ks.get("r"), dict) and ks["r"].get("a") == 1:
            props.add("rot")
        if isinstance(ks.get("p"), dict) and ks["p"].get("a") == 1:
            props.add("pos")
        if isinstance(ks.get("s"), dict) and ks["s"].get("a") == 1:
            props.add("scale")
        if isinstance(ks.get("o"), dict) and ks["o"].get("a") == 1:
            props.add("opacity")
    return props


def count_parent_links(layers):
    """How many layers have a parent."""
    return sum(1 for l in layers if l.get("parent") is not None)


def classify_motion(lottie, parts, props):
    """Heuristic motion category from keyframes + layer names + animated props."""
    layers = lottie.get("layers", [])
    names = [str(l.get("nm", "")).lower() for l in layers]
    joined = " ".join(names)

    # Name-based hints first
    for verb, hints in MOTION_VERBS.items():
        if any(h in joined for h in hints):
            return verb

    # Property-based fallback
    if "pos" in props and "rot" in props:
        return "walk" if any("leg" in n or "foot" in n for n in names) else "move"
    if "rot" in props and not props - {"rot"}:
        return "spin"
    if "scale" in props and not props - {"scale"}:
        return "pulse"
    if "opacity" in props and not props - {"opacity"}:
        return "fade"
    if not props:
        return "static"
    return "mixed"


def extract_facts(lottie, category):
    """Pull everything useful out of a Lottie file."""
    layers = lottie.get("layers", [])
    fr = lottie.get("fr", 30) or 30
    dur_frames = (lottie.get("op", 0) - lottie.get("ip", 0)) or 1
    dur_s = dur_frames / max(fr, 1)

    parts = detect_body_parts(layers)
    props = detect_animated_props(layers)
    parent_count = count_parent_links(layers)
    motion = classify_motion(lottie, parts, props)

    # Sanitize category for prompt use
    category_readable = category.replace("_", " ")

    return {
        "category": category_readable,
        "layer_count": len(layers),
        "duration_s": round(dur_s, 1),
        "fps": round(fr),
        "has_parent": parent_count > 0,
        "parent_count": parent_count,
        "parts": parts,            # Counter of body parts
        "props": sorted(props),    # list of animated properties
        "motion": motion,
        "w": lottie.get("w"),
        "h": lottie.get("h"),
    }


# ---------- Prompt generators (4 per file) ----------

def _parts_short(parts):
    """Short body-part description for prompts."""
    if not parts:
        return ""
    items = []
    for p, n in parts.most_common(5):
        if n == 1:
            items.append(p)
        else:
            items.append(f"{n} {p}s")
    return ", ".join(items)


def _props_short(props):
    if not props:
        return "no animation"
    names = {"rot": "rotation", "pos": "position",
             "scale": "scale", "opacity": "opacity"}
    return " + ".join(names.get(p, p) for p in props)


SUBJECT_BY_CATEGORY = {
    "character": ["character", "figure", "person"],
    "people":    ["person", "character", "figure"],
    "business":  ["business scene", "infographic", "business icon"],
    "emoji":     ["emoji", "smiley", "face"],
    "emotion":   ["emotional face", "expression"],
    "gesture":   ["gesture", "hand motion"],
    "animal":    ["animal", "creature"],
    "sport":     ["sports animation", "athlete"],
    "food":      ["food icon", "food animation"],
    "nature":    ["nature scene", "natural element"],
    "travel":    ["travel icon", "vehicle"],
    "medical":   ["medical icon", "health graphic"],
    "education": ["education icon", "learning graphic"],
    "ecommerce": ["ecommerce icon", "shopping graphic"],
    "technology":["tech icon", "device graphic"],
}


def _subject(facts, index=0):
    cat = facts["category"]
    options = SUBJECT_BY_CATEGORY.get(cat, [cat + " animation"])
    return options[index % len(options)]


def _motion_phrase(motion):
    table = {
        "walk":    "walking forward",
        "run":     "running",
        "wave":    "waving",
        "bounce":  "bouncing",
        "dance":   "dancing",
        "fly":     "flying",
        "swim":    "swimming",
        "blink":   "blinking",
        "breathe": "idling with breathing motion",
        "spin":    "spinning",
        "fade":    "fading in and out",
        "pulse":   "pulsing",
        "shake":   "shaking",
        "move":    "moving",
        "static":  "in a static pose",
        "mixed":   "animating",
    }
    return table.get(motion, motion)


# === Prompt 1: terse natural ===
def terse_natural(facts):
    subj = _subject(facts, 0)
    verb = _motion_phrase(facts["motion"])
    return f"{subj} {verb}"


# === Prompt 2: terse structural ===
def terse_structural(facts):
    n = facts["layer_count"]
    props = "+".join(facts["props"]) or "static"
    rig = " with rig" if facts["has_parent"] else ""
    return (f"{n}-layer {facts['category']} animation, "
            f"{props}, {facts['duration_s']}s @ {facts['fps']}fps{rig}")


# === Prompt 3: detailed natural ===
def detailed_natural(facts):
    subj = _subject(facts, 1)
    parts_desc = _parts_short(facts["parts"])
    verb = _motion_phrase(facts["motion"])
    n = facts["layer_count"]

    pieces = [f"A {subj}"]
    if parts_desc:
        pieces.append(f"with {parts_desc}")
    pieces.append(verb)
    pieces.append(f"over {facts['duration_s']} seconds")

    base = " ".join(pieces) + "."

    # Add prop detail
    if facts["props"]:
        base += f" The animation uses {_props_short(facts['props'])}."
    if facts["has_parent"]:
        base += f" {facts['parent_count']} layers are parented to a rig."
    return base


# === Prompt 4: detailed structural ===
def detailed_structural(facts):
    n = facts["layer_count"]
    parts_desc = _parts_short(facts["parts"]) or "generic layers"
    props = "+".join(facts["props"]) or "no animation"
    parent_desc = (f"{facts['parent_count']} layers linked via parent= hierarchy"
                   if facts["has_parent"] else "flat structure, no parent links")

    return (f"Generate a {n}-layer {facts['category']} animation with "
            f"{parts_desc}. Duration {facts['duration_s']}s at {facts['fps']}fps. "
            f"Animated properties: {props}. {parent_desc}. "
            f"Motion type: {facts['motion']}.")


PROMPT_FNS = [
    ("terse_natural",      terse_natural),
    ("terse_structural",   terse_structural),
    ("detailed_natural",   detailed_natural),
    ("detailed_structural",detailed_structural),
]


# ---------- Output stripping ----------

def strip_metadata(lottie):
    """Remove identifying metadata (so nothing traces back to IconScout)."""
    # Rename layer names to generic layer_N BUT preserve parent relationships
    # by doing this BEFORE we encode to AnimTOON (encoder uses ind field, not name)
    for i, l in enumerate(lottie.get("layers", [])):
        # Keep body-part hints but drop the numeric ID noise
        nm = str(l.get("nm", "")).lower()
        for part, hints in BODY_PART_HINTS.items():
            if any(h in nm for h in hints):
                l["nm"] = f"{part}_{i}"
                break
        else:
            l["nm"] = f"layer_{i}"
    for k in ["nm", "meta", "markers", "metadata"]:
        lottie.pop(k, None)
    return lottie


# ---------- Main pipeline ----------

def process_file(path, category, converter):
    """Return list of training records for one file, or [] if dropped."""
    try:
        lottie = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return []

    if not isinstance(lottie, dict) or "layers" not in lottie or "ip" not in lottie:
        return []

    # Extract facts BEFORE stripping (we want real layer names for body part detection)
    facts = extract_facts(lottie, category)

    # Strip metadata, then encode
    lottie = strip_metadata(lottie)
    try:
        animtoon = converter.convert(lottie)
    except Exception:
        return []

    tokens = count_tokens(animtoon)
    if tokens > MAX_OUTPUT_TOKENS:
        return []  # doesn't fit output budget

    # Generate 4 prompts + emit 4 records (same output for all 4)
    records = []
    for prompt_type, fn in PROMPT_FNS:
        try:
            prompt = fn(facts)
        except Exception:
            continue
        if not prompt or not prompt.strip():
            continue
        records.append({
            "prompt": prompt,
            "output": animtoon,
            "meta": {
                "category": category,
                "prompt_type": prompt_type,
                "layer_count": facts["layer_count"],
                "duration_s": facts["duration_s"],
                "fps": facts["fps"],
                "has_parent": facts["has_parent"],
                "parent_count": facts["parent_count"],
                "motion": facts["motion"],
                "animated_props": facts["props"],
                "token_count": tokens,
            }
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="process only N files per category (debug)")
    parser.add_argument("--category", default=None,
                        help="only process this category")
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    args = parser.parse_args()

    if not SCRAPED_ROOT.exists():
        print(f"No scraped data at {SCRAPED_ROOT}")
        return

    converter = LottieToAnimTOON()
    stats = Counter()
    drop_reasons = Counter()
    records_written = 0
    files_kept = 0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {SCRAPED_ROOT}")
    print(f"Output: {out_path}")
    print()

    with open(out_path, "w", encoding="utf-8") as fout:
        categories = sorted([d.name for d in SCRAPED_ROOT.iterdir() if d.is_dir()])
        if args.category:
            categories = [args.category]

        for cat in categories:
            cat_dir = SCRAPED_ROOT / cat
            files = [f for f in sorted(cat_dir.glob("*.json"))
                     if not f.name.endswith(".meta.json")]
            if args.limit:
                files = files[:args.limit]

            cat_files = 0
            cat_records = 0
            for f in files:
                stats["total"] += 1
                recs = process_file(f, cat, converter)
                if not recs:
                    drop_reasons["encode_fail_or_too_long"] += 1
                    continue
                cat_files += 1
                files_kept += 1
                for r in recs:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                    records_written += 1
                    cat_records += 1

            print(f"  [{cat:12s}] {cat_files:5d} files kept -> {cat_records:6d} records")

    print()
    print("=" * 60)
    print(f"Total files processed: {stats['total']}")
    print(f"Files kept:            {files_kept}")
    print(f"Files dropped:         {stats['total'] - files_kept}")
    for reason, n in drop_reasons.most_common():
        print(f"  {reason}: {n}")
    print(f"Training records:      {records_written}")
    print(f"Records per file:      {records_written / max(files_kept, 1):.1f}")
    print(f"Output file:           {out_path}")
    print(f"Output size:           {out_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
