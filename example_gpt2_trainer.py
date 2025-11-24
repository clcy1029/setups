#!/usr/bin/env python3
"""
train_gpt2.py

Production-ready GPT-2 (causal LM) training script using HuggingFace Transformers + Datasets.
Supports: resume, checkpointing, mixed precision, gradient accumulation, gradient clipping,
weight decay (AdamW), linear warmup + cosine/linear decay, logging (wandb optional).

Usage example (single-node multi-gpu with accelerate):
accelerate launch train_gpt2.py \
    --model_name_or_path gpt2 \
    --train_file data/train.txt \
    --output_dir ./checkpoints/gpt2-mini \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --max_seq_length 512 \
    --num_train_epochs 3 \
    --learning_rate 5e-4 \
    --weight_decay 0.01 \
    --warmup_steps 2000 \
    --fp16 True
"""

import os
import math
import argparse
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
# Optional logging
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

@dataclass
class Args:
    model_name_or_path: str = "gpt2"
    train_file: str = ""          # .txt or .jsonl (text field)
    output_dir: str = "./output"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 512
    num_train_epochs: int = 3
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    max_grad_norm: float = 1.0
    logging_steps: int = 100
    save_steps: int = 2000
    save_total_limit: int = 5
    fp16: bool = True
    seed: int = 42
    block_size: Optional[int] = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--report_to", type=str, default="none")  # "wandb" or "none"
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Args()
    # map CLI overrides:
    for k, v in vars(args).items():
        setattr(cfg, k, v)

    set_seed(cfg.seed)

    # --- tokenizer & model ---
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    # GPT-2 tokenizer may not have pad token; set it if missing
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token":"<|pad|>"})
    model_config = AutoConfig.from_pretrained(cfg.model_name_or_path)
    model = AutoModelForCausalLM.from_config(model_config)
    # We load pretrained weights if path given
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, config=model_config)
    except Exception as e:
        print("Warning: couldn't load pretrained weights: falling back to config init. Error:", e)

    # Resize token embeddings if tokenizer changed
    model.resize_token_embeddings(len(tokenizer))

    # --- dataset loading ---
    print("Loading dataset...")
    extension = cfg.train_file.split(".")[-1]
    if extension in ("txt",):
        dataset = load_dataset("text", data_files={"train": cfg.train_file}, keep_linebreaks=False)
        # dataset["train"] has column "text"
    elif extension in ("jsonl", "json"):
        dataset = load_dataset("json", data_files={"train": cfg.train_file})
    else:
        raise ValueError("Unsupported train_file extension, use .txt or .jsonl/.json")

    # Tokenize helper (we will use blockwise chunking)
    def tokenize_function(examples):
        # examples["text"] may be list
        return tokenizer(examples["text"], return_special_tokens_mask=False)

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset",
        num_proc=4
    )

    # Concatenate and group into blocks of block_size
    block_size = cfg.block_size or cfg.max_seq_length
    def group_texts(examples):
        # concatenate all input_ids
        concatenated = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)
        total_length = len(concatenated)
        # drop remainder to ensure exact multiples (common practice)
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        else:
            total_length = 0
        result = {
            "input_ids": [concatenated[i:i+block_size] for i in range(0, total_length, block_size)],
        }
        result["labels"] = [list(x) for x in result["input_ids"]]
        return result

    lm_dataset = tokenized["train"].map(
        group_texts,
        batched=True,
        batch_size=1000,
        desc=f"Grouping texts into chunks of {block_size}"
    )

    # Data collator for causal LM (labels are input_ids shifted inside model)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- Training Arguments (HuggingFace Trainer wrapper) ---
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=False,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        evaluation_strategy="no",
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        fp16=cfg.fp16,
        dataloader_num_workers=4,
        logging_steps=cfg.logging_steps,
        logging_strategy="steps",
        report_to=cfg.report_to if cfg.report_to in ("wandb","none") else "none",
        run_name=os.path.basename(cfg.output_dir),
        remove_unused_columns=False,
        max_grad_norm=cfg.max_grad_norm,
        push_to_hub=cfg.push_to_hub,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        data_collator=data_collator,
    )

    # Start training (supports resume)
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print("Training finished. Model & tokenizer saved to", cfg.output_dir)

if __name__ == "__main__":
    main()
