"""Launch AnimTOON training on SageMaker with managed spot instances.

Usage:
  python sagemaker/launch.py --base v4
  python sagemaker/launch.py --base v3
  python sagemaker/launch.py --base v4 --no-spot   # on-demand fallback

Monitors the job, streams logs, downloads adapter when done.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import boto3

# Load HF_TOKEN from .env
for line in Path(".env").read_text().splitlines():
    if line.startswith("HF_TOKEN="):
        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
        break

REGION = "us-east-1"
ACCOUNT = "698951528031"
ROLE_ARN = f"arn:aws:iam::{ACCOUNT}:role/AnimTOONSageMakerExecutionRole"
BUCKET = f"animtoon-sagemaker-{ACCOUNT}"
INSTANCE_TYPE = "ml.g6.xlarge"   # L4, 24 GB VRAM, ~$1.006/hr on-demand, ~$0.30 spot

BASE_MODEL_MAP = {
    "v3": "srk0102200/AnimTOON-3B",
    "v4": "srk0102200/AnimTOON-3B-v4",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", choices=["v3", "v4"], default="v4")
    p.add_argument("--no-spot", action="store_true")
    p.add_argument("--max-wait-hours", type=float, default=6.0)
    p.add_argument("--max-run-hours", type=float, default=5.0)
    args = p.parse_args()

    base_model = BASE_MODEL_MAP[args.base]
    job_name = f"animtoon-3b-{args.base}1-{int(time.time())}"
    output_path = f"s3://{BUCKET}/training-jobs/{job_name}"

    # Upload training script + requirements to S3
    s3 = boto3.client("s3", region_name=REGION)
    src_dir = Path(__file__).parent / "_sagemaker_source"
    code_prefix = f"code/{job_name}"

    for local_file in ["train.py", "requirements.txt"]:
        local_path = src_dir / local_file
        key = f"{code_prefix}/{local_file}"
        print(f"Uploading s3://{BUCKET}/{key}")
        s3.upload_file(str(local_path), BUCKET, key)

    # We need to create a tar.gz of the source for SageMaker
    import tarfile
    import io
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        tar.add(src_dir / "train.py", arcname="train.py")
        tar.add(src_dir / "requirements.txt", arcname="requirements.txt")
    tar_buffer.seek(0)
    source_key = f"{code_prefix}/sourcedir.tar.gz"
    s3.put_object(Bucket=BUCKET, Key=source_key, Body=tar_buffer.getvalue())
    source_s3 = f"s3://{BUCKET}/{source_key}"
    print(f"Uploaded source: {source_s3}")

    # Launch training job via sagemaker SDK
    from sagemaker.huggingface import HuggingFace

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("WARNING: HF_TOKEN not set in environment")

    hyperparameters = {
        "base-model": base_model,
        "data-repo": "srk0102200/animtoon-iconscout-v4",
        "data-file": "iconscout_training_v4.jsonl",
        "hf-token": hf_token,
        "epochs": 2,
        "batch-size": 4,
        "grad-accum": 4,
        "learning-rate": 2e-4,
        "lora-r": 16,
        "lora-alpha": 32,
        "lora-dropout": 0.05,
        "max-seq-length": 2048,
        "save-steps": 500,
        "eval-steps": 500,
        "logging-steps": 25,
    }

    spot_settings = {}
    if not args.no_spot:
        spot_settings = {
            "use_spot_instances": True,
            "max_wait": int(args.max_wait_hours * 3600),
            "checkpoint_s3_uri": f"s3://{BUCKET}/checkpoints/{job_name}",
            "checkpoint_local_path": "/opt/ml/checkpoints",
        }
        print(f"SPOT ENABLED: max_wait={args.max_wait_hours}h")
    else:
        print("ON-DEMAND (no spot)")

    estimator = HuggingFace(
        entry_point="train.py",
        source_dir=str(src_dir),
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        role=ROLE_ARN,
        transformers_version="4.46",
        pytorch_version="2.3",
        py_version="py311",
        hyperparameters=hyperparameters,
        output_path=output_path,
        base_job_name=f"animtoon-{args.base}1",
        max_run=int(args.max_run_hours * 3600),
        **spot_settings,
    )

    print(f"\n{'='*60}")
    print(f"Launching SageMaker training job: {job_name}")
    print(f"  Base model:    {base_model}")
    print(f"  Instance:      {INSTANCE_TYPE}")
    print(f"  Spot:          {not args.no_spot}")
    print(f"  Role:          {ROLE_ARN}")
    print(f"  Output:        {output_path}")
    print(f"{'='*60}\n")

    estimator.fit(wait=False)
    launched_name = estimator.latest_training_job.name
    print(f"\n[OK] Job launched: {launched_name}")
    print(f"  Monitor: aws sagemaker describe-training-job --training-job-name {launched_name}")
    print(f"  Console: https://{REGION}.console.aws.amazon.com/sagemaker/home?region={REGION}#/jobs/{launched_name}")
    print(f"\n  Model will be at: {output_path}/{launched_name}/output/model.tar.gz")


if __name__ == "__main__":
    main()
