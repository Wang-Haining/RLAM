"""
This module performs supervised fine-tuning on the Flan T5-XL model (2.85B) using the
Simplified Abstract Simplification (SAS) dataset.
"""

__author__ = "hw56@indiana.edu"
__version__ = "0.0.1"
__license__ = "0BSD"

import os

import torch
import wandb
from datasets import DatasetDict, load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from utils import (
    DATASET_PATH,
    PROJECT_NAME,
    RESPONSE_TEMP,
    SEED,
    SEQ2SEQ_MODEL_NAME,
    T5_MAX_INPUT_LEN,
    T5_MAX_OUTPUT_LEN,
    TASK_PREFIX,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
run_name = f'sft_{SEQ2SEQ_MODEL_NAME.split("/")[-1]}'
tokenizer = AutoTokenizer.from_pretrained(SEQ2SEQ_MODEL_NAME, padding_side="right")


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
    torch.manual_seed(SEED + 21)

    dataset = load_from_disk(DATASET_PATH)
    train_dataset = dataset["train"].map(
        lambda batch: preprocess_function(batch, tokenizer), batched=True, num_proc=2
    )
    val_dataset = dataset["validation"].map(
        lambda batch: preprocess_function(batch, tokenizer), batched=True, num_proc=2
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        SEQ2SEQ_MODEL_NAME, torch_dtype=torch.bfloat16
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=f"ckpts/{run_name}",
        overwrite_output_dir=False,
        num_train_epochs=10.0,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
