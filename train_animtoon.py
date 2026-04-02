"""
AnimTOON Training Script

Fine-tunes Qwen2.5-3B-Instruct with LoRA on AnimTOON training pairs.
Text-only fine-tuning (no vision needed for text→AnimTOON generation).

Usage:
    python train_animtoon.py                              # default settings
    python train_animtoon.py --data animtoon_train.jsonl   # custom data
    python train_animtoon.py --epochs 5 --lr 1e-4          # custom hyperparams
    python train_animtoon.py --resume ./models/checkpoint-500  # resume

Requirements:
    pip install torch transformers peft datasets accelerate bitsandbytes
    Optional: pip install wandb
"""

import json
import os
import sys
import argparse

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType


# ─── DATA LOADING ────────────────────────────────────────────────────

def load_training_data(jsonl_path: str) -> Dataset:
    """Load AnimTOON training pairs from JSONL file."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(records):,} training pairs from {jsonl_path}")
    return Dataset.from_list(records)


def format_chat(example: dict, tokenizer) -> dict:
    """Format instruction/output as chat messages and tokenize."""
    messages = [
        {
            "role": "system",
            "content": "You are AnimTOON, a Lottie animation generator. "
                       "Given a description, output the animation in AnimTOON format."
        },
        {
            "role": "user",
            "content": f"Generate AnimTOON animation: {example['instruction']}"
        },
        {
            "role": "assistant",
            "content": example['output']
        },
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def tokenize_fn(examples: dict, tokenizer, max_length: int) -> dict:
    """Tokenize formatted text for causal LM training."""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    # For causal LM, labels = input_ids (shifted internally by the model)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# ─── MODEL SETUP ─────────────────────────────────────────────────────

def setup_model_and_tokenizer(
    model_name: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    load_in_4bit: bool = False,
):
    """Load base model with LoRA adapter."""
    print(f"Loading model: {model_name}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config for 4-bit (saves VRAM)
    quantization_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if not load_in_4bit else None,
        device_map="auto",
    )

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ─── TRAINING ────────────────────────────────────────────────────────

def train(
    data_path: str = "animtoon_train.jsonl",
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    output_dir: str = "./models/animtoon-3b",
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 2e-4,
    max_length: int = 1024,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    load_in_4bit: bool = False,
    save_steps: int = 500,
    resume_from: str = None,
    use_wandb: bool = True,
):
    """Full training pipeline."""
    print("=" * 50)
    print("=== AnimTOON Training ===")
    print(f"  Model: {model_name}")
    print(f"  Data: {data_path}")
    print(f"  Output: {output_dir}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size} x {grad_accum} grad accum = {batch_size * grad_accum} effective")
    print(f"  Learning rate: {lr}")
    print(f"  Max length: {max_length}")
    print(f"  LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"  4-bit quantization: {load_in_4bit}")
    print("=" * 50)
    print()

    # Check CUDA
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected. Training will be very slow.")
    print()

    # Load data
    dataset = load_training_data(data_path)
    if len(dataset) == 0:
        print("ERROR: No training data found.")
        sys.exit(1)

    # Load model
    model, tokenizer = setup_model_and_tokenizer(
        model_name, lora_r, lora_alpha, lora_dropout, load_in_4bit
    )

    # Format and tokenize
    print("Formatting and tokenizing...")
    dataset = dataset.map(
        lambda ex: format_chat(ex, tokenizer),
        desc="Formatting chat",
    )
    dataset = dataset.map(
        lambda ex: tokenize_fn(ex, tokenizer, max_length),
        remove_columns=["instruction", "output", "text"],
        desc="Tokenizing",
    )

    # Show sample lengths
    lengths = [len(x) for x in dataset["input_ids"]]
    print(f"  Token lengths: min={min(lengths)}, avg={sum(lengths)//len(lengths)}, "
          f"max={max(lengths)}, truncated={sum(1 for l in lengths if l >= max_length)}")
    print()

    # Wandb setup
    report_to = "none"
    if use_wandb:
        try:
            import wandb
            wandb.init(project="animtoon", name=f"animtoon-{model_name.split('/')[-1]}")
            report_to = "wandb"
            print("Wandb logging enabled")
        except (ImportError, Exception):
            print("Wandb not available, logging to tensorboard")
            report_to = "tensorboard"

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=3,
        report_to=report_to,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        optim="adamw_torch",
        gradient_checkpointing=True,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    print("Starting training...")
    if resume_from:
        print(f"Resuming from: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    # Save final model
    print(f"\nSaving final model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save LoRA adapter separately
    adapter_dir = os.path.join(output_dir, "lora_adapter")
    model.save_pretrained(adapter_dir)
    print(f"LoRA adapter saved to {adapter_dir}")

    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AnimTOON model")
    parser.add_argument("--data", default="animtoon_train.jsonl",
                        help="Training data JSONL path")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                        help="Base model name")
    parser.add_argument("--output", default="./models/animtoon-3b",
                        help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--4bit", action="store_true", dest="load_4bit",
                        help="Use 4-bit quantization (saves VRAM)")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    args = parser.parse_args()

    train(
        data_path=args.data,
        model_name=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        load_in_4bit=args.load_4bit,
        save_steps=args.save_steps,
        resume_from=args.resume,
        use_wandb=not args.no_wandb,
    )
