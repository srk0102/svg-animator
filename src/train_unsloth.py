"""
AnimTOON Training Script (Unsloth + Docker)

Run inside Unsloth Docker container:
    docker run --gpus all -it -v D:\svg-animator:/workspace unsloth/unsloth bash
    cd /workspace
    python src/train_unsloth.py --data data/animtoon_train_10k.jsonl

Unsloth benefits:
    - 2x faster training
    - 70% less VRAM
    - Lower GPU temps
"""

import json
import os
import sys
import argparse

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer


# ─── DATA ────────────────────────────────────────────────────────────

def load_training_data(jsonl_path: str) -> Dataset:
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                # Support both 'instruction' and 'input' field names
                if rec.get("input") and not rec.get("instruction"):
                    rec["instruction"] = rec["input"]
                if rec.get("instruction") and rec.get("output"):
                    records.append(rec)
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(records):,} training pairs from {jsonl_path}")
    return Dataset.from_list(records)


def format_chat(example: dict, tokenizer) -> dict:
    messages = [
        {"role": "user", "content": f"Generate AnimTOON animation: {example['instruction']}"},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def tokenize_fn(examples: dict, tokenizer, max_length: int) -> dict:
    tokenized = tokenizer(examples["text"], truncation=True, max_length=max_length, padding=False)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# ─── TRAIN ───────────────────────────────────────────────────────────

def train(
    data_path: str = "data/animtoon_train.jsonl",
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    output_dir: str = "models/animtoon-3b",
    epochs: int = 3,
    batch_size: int = 1,
    grad_accum: int = 16,
    lr: float = 2e-4,
    max_length: int = 1024,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    save_steps: int = 200,
    eval_steps: int = 200,
    resume_from: str = None,
):
    effective_bs = batch_size * grad_accum
    print("=" * 50)
    print("=== AnimTOON Training (Unsloth) ===")
    print(f"  Model:          {model_name}")
    print(f"  Data:           {data_path}")
    print(f"  Output:         {output_dir}")
    print(f"  Epochs:         {epochs}")
    print(f"  Batch size:     {batch_size} x {grad_accum} = {effective_bs} effective")
    print(f"  Learning rate:  {lr}")
    print(f"  Max length:     {max_length}")
    print(f"  LoRA:           r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"  Save/eval every {save_steps} steps")
    print("=" * 50)
    print()

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu} ({vram:.1f} GB VRAM)")
    print()

    # Load data
    dataset = load_training_data(data_path)
    if len(dataset) == 0:
        print("ERROR: No training data found.")
        sys.exit(1)

    # Load model with Unsloth
    print(f"Loading model with Unsloth: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_length,
        load_in_4bit=True,
        dtype=None,
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format and tokenize
    print("Formatting and tokenizing...")
    dataset = dataset.map(lambda ex: format_chat(ex, tokenizer), desc="Formatting chat")
    dataset = dataset.map(
        lambda ex: tokenize_fn(ex, tokenizer, max_length),
        remove_columns=dataset.column_names, desc="Tokenizing",
    )

    lengths = [len(x) for x in dataset["input_ids"]]
    print(f"  Token lengths: min={min(lengths)}, avg={sum(lengths)//len(lengths)}, "
          f"max={max(lengths)}, truncated={sum(1 for l in lengths if l >= max_length)}")
    print()

    # 95/5 split
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset, eval_dataset = split["train"], split["test"]
    print(f"  Train: {len(train_dataset):,}  |  Eval: {len(eval_dataset):,}")
    print()

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        bf16=True,
        max_grad_norm=1.0,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        optim="adamw_8bit",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # Train
    print("Starting training...")
    if resume_from:
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    # Save
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    print(f"\nSaving final model to {final_dir}")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"LoRA adapter + tokenizer saved to {final_dir}")
    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AnimTOON (Unsloth)")
    parser.add_argument("--data", default="data/animtoon_train.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output", default="models/animtoon-3b")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    train(
        data_path=args.data, model_name=args.model, output_dir=args.output,
        epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum,
        lr=args.lr, max_length=args.max_length, lora_r=args.lora_r,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        save_steps=args.save_steps, eval_steps=args.eval_steps, resume_from=args.resume,
    )
