"""AnimTOON LoRA training script for SageMaker Training Jobs.

Runs on a single GPU (ml.g6.xlarge L4 in our case).
Downloads base model + training data from HuggingFace, trains LoRA, saves
adapter to /opt/ml/model/ (SageMaker auto-uploads this to S3 on job completion).

Environment variables passed in via SageMaker hyperparameters:
  BASE_MODEL      HF repo id of base model (e.g. srk0102200/AnimTOON-3B-v4)
  DATA_REPO       HF dataset repo (e.g. srk0102200/animtoon-iconscout-v4)
  DATA_FILE       filename in dataset repo (e.g. iconscout_training_v4.jsonl)
  HF_TOKEN        HuggingFace token for private repos
  EPOCHS, BATCH_SIZE, GRAD_ACCUM, LEARNING_RATE, etc.
"""
import argparse
import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", type=str, required=True)
    p.add_argument("--data-repo", type=str, required=True)
    p.add_argument("--data-file", type=str, default="iconscout_training_v4.jsonl")
    p.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--logging-steps", type=int, default=25)

    # SageMaker-injected paths
    p.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    p.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    return p.parse_args()


def main():
    args = parse_args()
    log.info(f"Starting AnimTOON training")
    log.info(f"  base_model = {args.base_model}")
    log.info(f"  data_repo  = {args.data_repo}")
    log.info(f"  epochs     = {args.epochs}")
    log.info(f"  batch      = {args.batch_size} x grad_accum {args.grad_accum} = eff {args.batch_size * args.grad_accum}")
    log.info(f"  lr         = {args.learning_rate}")
    log.info(f"  model_dir  = {args.model_dir}")

    import torch
    log.info(f"torch {torch.__version__}")
    log.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        log.info(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
    else:
        raise RuntimeError("No CUDA GPU detected!")

    # HF login
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        from huggingface_hub import login
        login(token=args.hf_token)
        log.info("HF login successful")

    # Download training data
    from huggingface_hub import hf_hub_download
    log.info(f"Downloading {args.data_file} from {args.data_repo} ...")
    data_path = hf_hub_download(
        repo_id=args.data_repo,
        filename=args.data_file,
        repo_type="dataset",
        token=args.hf_token,
    )
    log.info(f"Data downloaded to {data_path}")

    records = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            records.append({"prompt": r["prompt"], "output": r["output"]})
    log.info(f"Loaded {len(records)} training records")
    log.info(f"Sample prompt: {records[0]['prompt'][:100]}")

    # Load model (4-bit QLoRA)
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    log.info(f"Loading {args.base_model} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=args.hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        padding_side="left",
        token=args.hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    log.info(f"Model loaded. Tokenizer vocab: {len(tokenizer)}")

    # Attach LoRA
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Format + tokenize
    from datasets import Dataset

    def format_record(r):
        messages = [
            {"role": "user",      "content": f"Generate AnimTOON animation: {r['prompt']}"},
            {"role": "assistant", "content": r["output"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}

    dataset = Dataset.from_list(records)
    dataset = dataset.map(format_record, remove_columns=dataset.column_names)
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    log.info(f"Train: {len(train_dataset)}  Eval: {len(eval_dataset)}")

    # SFTTrainer
    from trl import SFTTrainer, SFTConfig
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        args=SFTConfig(
            output_dir=args.output_data_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type="cosine",
            weight_decay=args.weight_decay,
            max_grad_norm=1.0,
            optim="paged_adamw_8bit",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=args.logging_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=2,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",
            dataset_text_field="text",
            seed=42,
        ),
    )
    log.info("Trainer ready. Starting training...")
    stats = trainer.train()
    log.info(f"Training complete. Final loss: {stats.training_loss:.4f}")

    # Save LoRA adapter + tokenizer to /opt/ml/model/ so SageMaker uploads to S3
    log.info(f"Saving LoRA adapter to {args.model_dir}")
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

    # Quick inference sanity check
    log.info("Inference sanity check:")
    model.eval()
    test_prompts = [
        "a red circle pulsing in the center",
        "a businessman waving hello",
        "a cat with head, body, 4 legs, tail",
    ]
    for p in test_prompts:
        messages = [{"role": "user", "content": f"Generate AnimTOON animation: {p}"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=512, temperature=0.5,
                top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )
        result = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        log.info(f"\n=== {p} ===")
        log.info(result[:400])

    log.info("All done. LoRA adapter saved to /opt/ml/model/ — SageMaker will upload to S3.")


if __name__ == "__main__":
    main()
