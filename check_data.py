"""Check training data for contamination."""
import json

contaminated = 0
clean = 0
total = 0
examples_bad = []
examples_good = []

with open("data/animtoon_train_10k.jsonl", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line.strip())
        out = rec["output"]
        total += 1

        # Check for raw JSON contamination
        has_json_blob = '{"' in out or "'s':" in out or '"ks"' in out
        starts_with_anim = out.strip().startswith("anim ")
        has_arrow = "\u2192" in out

        if has_json_blob:
            contaminated += 1
            if len(examples_bad) < 3:
                examples_bad.append(out[:300])
        elif starts_with_anim and has_arrow:
            clean += 1
            if len(examples_good) < 2:
                examples_good.append(out[:300])
        else:
            clean += 1

print(f"Total samples: {total}")
print(f"Clean AnimTOON: {clean} ({clean/total*100:.1f}%)")
print(f"Contaminated (has JSON): {contaminated} ({contaminated/total*100:.1f}%)")
print()

if examples_bad:
    print("=== CONTAMINATED EXAMPLES ===")
    for i, ex in enumerate(examples_bad):
        print(f"[BAD {i+1}] {ex}")
        print()

print("=== CLEAN EXAMPLES ===")
for i, ex in enumerate(examples_good):
    print(f"[GOOD {i+1}] {ex}")
    print()
