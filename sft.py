"""
Get supervised finetuned Gemma 2B from SAS.
"""

__author__ = "hw56@indiana.edu"

import os

import torch
import wandb
from datasets import load_from_disk
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          TrainingArguments,
                          EarlyStoppingCallback)

from trl import SFTTrainer

from utils import (DATASET_PATH, SEED, PROJECT_NAME, MODEL_NAME,
                   RESPONSE_TEMP, TASK_PREFIX)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
run_name = f'sft_{MODEL_NAME.split("/")[-1]}'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")


def formatting_func(example):
    output_texts = []
    for i in range(len(example["source"])):
        text = (TASK_PREFIX +
                f"{example['source'][i]}{RESPONSE_TEMP} {example['target'][i]}")
        output_texts.append(text)

    return output_texts


if __name__ == "__main__":

    torch.manual_seed(SEED + 21)

    dataset = load_from_disk(DATASET_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                                 torch_dtype=torch.bfloat16)

    training_args = TrainingArguments(
        output_dir=f"ckpts/{run_name}",
        overwrite_output_dir=False,
        num_train_epochs=10.0,
        do_train=True,
        do_eval=True,
        do_predict=True,
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
    wandb.init(project=PROJECT_NAME,
               name=run_name,
               config=training_args)

    trainer = SFTTrainer(
        model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        formatting_func=formatting_func,
        max_seq_length=1024,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
