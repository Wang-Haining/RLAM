"""
This module performs supervised finetuning on the FLAN-T5-XL using the
Scientific Abstract-Significance Statement dataset (SASS). It maps scientific
abstracts with their simplified versions using a straightforward template.
"""

__author__ = "hw56@indiana.edu"
__version__ = "0.0.1"
__license__ = "0BSD"

import os

import torch
import wandb
from datasets import DatasetDict, load_from_disk
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, EarlyStoppingCallback,
                          Trainer, TrainingArguments)

from utils import (FLANT5_XL, DATASET_PATH, PROJECT_NAME, RESPONSE_TEMP, SEED,
                   TASK_PREFIX, CKPTS_DIR)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
run_name = f'sft_{FLANT5_XL.split("/")[-1]}'
tokenizer = AutoTokenizer.from_pretrained(FLANT5_XL, padding_side="right")


def formatting_func(examples, tokenizer):
    inputs = [TASK_PREFIX + text + RESPONSE_TEMP + " " for text in examples["source"]]
    targets = [text for text in examples["target"]]

    model_inputs = tokenizer(
        inputs,
        max_length=512,  # only 27 (out of 3030) longer than 512
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    labels = tokenizer(
        targets,
        max_length=256,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    model_inputs["labels"] = labels.input_ids

    return model_inputs


if __name__ == "__main__":
    torch.manual_seed(SEED + 8046251)

    dataset = load_from_disk(DATASET_PATH)
    train_dataset = dataset["train"].map(
        lambda batch: formatting_func(batch, tokenizer), batched=True)
    val_dataset = dataset["validation"].map(
        lambda batch: formatting_func(batch, tokenizer), batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(FLANT5_XL,
                                                  torch_dtype=torch.bfloat16)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=f"CKPTS_DIR/{run_name}",
        overwrite_output_dir=False,
        num_train_epochs=50.0,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
