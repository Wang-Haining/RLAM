"""
This module performs supervised finetuning on a Flan-T5 base using the Scientific
Abstract-Significance Statement dataset (SASS). It concatenates scientific abstracts
with their simplified versions using a straightforward template.
"""

__author__ = "hw56@indiana.edu"
__version__ = "0.0.1"
__license__ = "0BSD"

import argparse
import os

import torch
import wandb
from datasets import load_from_disk
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, EarlyStoppingCallback,
                          Trainer, TrainingArguments, set_seed)

from utils import (CKPTS_DIR, DATASET_PATH, PROJECT_NAME, RESPONSE_TEMP, SEED,
                   TASK_PREFIX)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
T5_MODEL_NAME = "google/flan-t5-xl"
T5_MAX_INPUT_LEN = 512  # max length == 661
T5_MAX_OUTPUT_LEN = 275
run_name = f'sft_{T5_MODEL_NAME.split("/")[-1]}'
tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_NAME, padding_side="right")


def preprocess_function(examples, tokenizer):
    inputs = [TASK_PREFIX + text + RESPONSE_TEMP for text in examples["source"]]
    targets = [text for text in examples["target"]]

    model_inputs = tokenizer(
        inputs,
        max_length=T5_MAX_INPUT_LEN,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    labels = tokenizer(
        targets,
        max_length=T5_MAX_OUTPUT_LEN,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    model_inputs["labels"] = labels.input_ids

    return model_inputs


if __name__ == "__main__":
    set_seed(SEED + 2122)
    parser = argparse.ArgumentParser(description="Supervise Fine-tuning with google/flan-t5-xl.")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    args = parser.parse_args()

    dataset = load_from_disk(DATASET_PATH)
    train_dataset = dataset["train"].map(
        lambda batch: preprocess_function(batch, tokenizer), batched=True)
    val_dataset = dataset["validation"].map(
        lambda batch: preprocess_function(batch, tokenizer), batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        T5_MODEL_NAME, torch_dtype=torch.bfloat16
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=f"{CKPTS_DIR}/{run_name}",
        overwrite_output_dir=True,
        num_train_epochs=50.0,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        lr_scheduler_type='constant_with_warmup',
        warmup_steps=50,
        weight_decay=1e-1,
        logging_steps=20,
        eval_steps=20,
        bf16=True,
        report_to="wandb",
        load_best_model_at_end=True,
        save_steps=20,
        save_total_limit=3,
        remove_unused_columns=True,
    )
    wandb.init(project=PROJECT_NAME, name=run_name, config=training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
