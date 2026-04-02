"""
AnimTOON Dataset Pipeline

Converts MMLottie-2M Lottie JSONs to AnimTOON format training pairs.

Supports two modes:
  1. Local parquet files (fast, recommended):
     python dataset_pipeline.py --parquet ./data/*.parquet
     python dataset_pipeline.py --parquet ./Lottie/train/0000.parquet

  2. HuggingFace streaming (slow, needs network):
     python dataset_pipeline.py --hf --limit 1000

Download parquet files from:
  https://huggingface.co/datasets/OmniLottie/MMLottie-2M/tree/main/Lottie/train
"""

import json
import os
import sys
import glob
import time
import argparse
from typing import Optional, Iterator

from toon_animator import LottieToAnimTOON, count_tokens


def build_instruction(sample: dict) -> str:
    """Build text instruction from available description fields.
    Priority: detail > desc_en > motion_caption > keywords_en."""
    desc = (sample.get('detail') or '').strip()
    if desc:
        return desc

    desc = (sample.get('desc_en') or '').strip()
    if desc:
        return desc

    desc = (sample.get('motion_caption') or '').strip()
    if desc:
        return desc

    kw = (sample.get('keywords_en') or '').strip()
    if kw:
        return f"Create a Lottie animation with: {kw}"

    return "Create a Lottie animation"


def process_sample(sample: dict, converter: LottieToAnimTOON) -> Optional[dict]:
    """Process a single sample. Returns training pair dict or None."""
    lottie_str = sample.get('lottie_json', '')
    if not lottie_str:
        return None

    try:
        lottie_json = json.loads(lottie_str) if isinstance(lottie_str, str) else lottie_str
    except (json.JSONDecodeError, TypeError):
        return None

    layers = lottie_json.get('layers', [])
    if not layers:
        return None

    try:
        animtoon_text = converter.convert(lottie_json)
    except Exception:
        return None

    if len(animtoon_text.strip()) < 20:
        return None

    instruction = build_instruction(sample)
    original_tokens = count_tokens(lottie_str if isinstance(lottie_str, str) else json.dumps(lottie_str))
    animtoon_tokens = count_tokens(animtoon_text)

    return {
        'id': sample.get('id', ''),
        'instruction': instruction,
        'output': animtoon_text,
        'original_tokens': original_tokens,
        'animtoon_tokens': animtoon_tokens,
        'num_layers': len(layers),
    }


# ─── DATA LOADING ────────────────────────────────────────────────────

def iter_parquet_files(parquet_paths: list, limit: int = 0) -> Iterator[dict]:
    """Iterate rows from one or more parquet files."""
    import pyarrow.parquet as pq

    count = 0
    for path in sorted(parquet_paths):
        print(f"  Reading: {os.path.basename(path)}")
        table = pq.read_table(path, columns=[
            'id', 'detail', 'desc_en', 'keywords_en',
            'motion_caption', 'lottie_json'
        ])
        for i in range(table.num_rows):
            if 0 < limit <= count:
                return
            row = {col: table.column(col)[i].as_py() for col in table.column_names}
            yield row
            count += 1


def iter_hf_api(subset: str = 'Lottie', limit: int = 100) -> Iterator[dict]:
    """Fetch rows via HuggingFace datasets-server REST API.
    No dependencies needed beyond urllib. Fast for small batches."""
    import urllib.request
    import urllib.parse

    base_url = "https://datasets-server.huggingface.co/rows"
    batch_size = 100  # API max per request
    offset = 0
    count = 0
    target = limit if limit > 0 else 999999999

    while count < target:
        length = min(batch_size, target - count)
        params = urllib.parse.urlencode({
            'dataset': 'OmniLottie/MMLottie-2M',
            'config': subset,
            'split': 'train',
            'offset': offset,
            'length': length,
        })
        url = f"{base_url}?{params}"

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            print(f"  API error at offset {offset}: {e}")
            break

        rows = data.get('rows', [])
        if not rows:
            break

        for row_obj in rows:
            if count >= target:
                return
            yield row_obj.get('row', {})
            count += 1

        offset += len(rows)
        if len(rows) < length:
            break  # No more data


def iter_huggingface(subset: str = 'Lottie', limit: int = 0) -> Iterator[dict]:
    """Stream from HuggingFace datasets library (needs pip install datasets)."""
    from datasets import load_dataset

    ds = load_dataset("OmniLottie/MMLottie-2M", subset, split="train", streaming=True)
    count = 0
    for sample in ds:
        if 0 < limit <= count:
            return
        yield sample
        count += 1


# ─── MAIN PIPELINE ───────────────────────────────────────────────────

def run_pipeline(
    data_iter: Iterator[dict],
    limit: int,
    output_file: str,
    batch_log_every: int = 100,
):
    """Process samples from iterator, write training pairs to JSONL."""
    print()
    print("Processing samples...")
    converter = LottieToAnimTOON()

    total = 0
    success = 0
    errors = 0
    skipped_no_desc = 0
    skipped_no_json = 0
    skipped_convert_fail = 0

    sum_original_tokens = 0
    sum_animtoon_tokens = 0
    sum_layers = 0
    max_animtoon_tokens = 0
    min_animtoon_tokens = float('inf')

    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or '.', exist_ok=True)
    t_start = time.time()

    with open(output_file, 'w') as f_out:
        for sample in data_iter:
            if 0 < limit <= total:
                break
            total += 1

            if not sample.get('lottie_json'):
                skipped_no_json += 1
                errors += 1
                continue

            result = process_sample(sample, converter)

            if result is None:
                skipped_convert_fail += 1
                errors += 1
                continue

            if result['instruction'] == "Create a Lottie animation":
                skipped_no_desc += 1

            training_pair = {
                'instruction': result['instruction'],
                'output': result['output'],
            }
            f_out.write(json.dumps(training_pair, ensure_ascii=False) + '\n')
            success += 1

            sum_original_tokens += result['original_tokens']
            sum_animtoon_tokens += result['animtoon_tokens']
            sum_layers += result['num_layers']
            max_animtoon_tokens = max(max_animtoon_tokens, result['animtoon_tokens'])
            min_animtoon_tokens = min(min_animtoon_tokens, result['animtoon_tokens'])

            if success % batch_log_every == 0:
                elapsed = time.time() - t_start
                rate = total / elapsed if elapsed > 0 else 0
                print(f"  [{total:,} processed, {success:,} saved, "
                      f"{errors:,} errors] {rate:.0f} samples/sec")

    elapsed = time.time() - t_start

    # ── Statistics ──
    print()
    print("=" * 50)
    print("=== Pipeline Complete ===")
    print(f"  Time elapsed: {elapsed:.1f}s")
    print(f"  Samples processed: {total:,}")
    print(f"  Successfully converted: {success:,}")
    print(f"  Errors/skipped: {errors:,}")
    print(f"    - No lottie_json: {skipped_no_json:,}")
    print(f"    - Conversion failed: {skipped_convert_fail:,}")
    print(f"    - No description (still saved): {skipped_no_desc:,}")
    print()

    if success > 0:
        avg_orig = sum_original_tokens / success
        avg_anim = sum_animtoon_tokens / success
        avg_layers = sum_layers / success
        reduction = (1 - avg_anim / avg_orig) * 100 if avg_orig > 0 else 0

        print("=== Token Statistics ===")
        print(f"  Avg original tokens:  {avg_orig:,.0f}")
        print(f"  Avg AnimTOON tokens:  {avg_anim:,.0f}")
        print(f"  Min AnimTOON tokens:  {min_animtoon_tokens:,}")
        print(f"  Max AnimTOON tokens:  {max_animtoon_tokens:,}")
        print(f"  Avg token reduction:  {reduction:.1f}%")
        print(f"  Avg layers per anim:  {avg_layers:.1f}")
        print()

    out_size = os.path.getsize(output_file)
    print(f"  Output: {output_file} ({out_size // 1024:,} KB)")
    print(f"  Training pairs: {success:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AnimTOON Dataset Pipeline")
    parser.add_argument('--parquet', nargs='+',
                        help='Local parquet file(s) or glob pattern')
    parser.add_argument('--api', action='store_true',
                        help='Fetch via HuggingFace REST API (no deps, recommended)')
    parser.add_argument('--hf', action='store_true',
                        help='Stream via HuggingFace datasets library')
    parser.add_argument('--subset', default='Lottie',
                        choices=['Lottie', 'Lottie_SVG'],
                        help='HF subset (default: Lottie)')
    parser.add_argument('--limit', type=int, default=100,
                        help='Max samples to process (0 = all)')
    parser.add_argument('--output', default='animtoon_train.jsonl',
                        help='Output JSONL file path')
    args = parser.parse_args()

    print("=== AnimTOON Dataset Pipeline ===")
    print(f"Limit: {args.limit if args.limit > 0 else 'ALL'}")
    print(f"Output: {args.output}")

    if args.parquet:
        files = []
        for p in args.parquet:
            files.extend(glob.glob(p))
        files = [f for f in files if f.endswith('.parquet')]
        if not files:
            print(f"ERROR: No parquet files found matching: {args.parquet}")
            sys.exit(1)
        print(f"Source: {len(files)} local parquet file(s)")
        data_iter = iter_parquet_files(files, args.limit)
    elif args.api:
        print(f"Source: HuggingFace REST API ({args.subset})")
        data_iter = iter_hf_api(args.subset, args.limit)
    elif args.hf:
        print(f"Source: HuggingFace datasets streaming ({args.subset})")
        data_iter = iter_huggingface(args.subset, args.limit)
    else:
        # Default to API mode — no extra deps needed
        print(f"Source: HuggingFace REST API ({args.subset}) [default]")
        data_iter = iter_hf_api(args.subset, args.limit)

    run_pipeline(data_iter, args.limit, args.output)
