"""
This module performs supervised fine-tuning on the OLMo 1B and Gemma 2B using the
Scientific Abstract-Significance Statement dataset (SASS). It concatenates scientific
abstracts with their simplified versions using a straightforward template.
"""

__author__ = "hw56@indiana.edu"
__version__ = "0.0.1"
__license__ = "0BSD"

import argparse
import os
from typing import List

import torch
import wandb
from datasets import DatasetDict, load_from_disk
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          EarlyStoppingCallback, TrainingArguments)
from trl import SFTTrainer, set_seed
from peft import LoraConfig, get_peft_model

from utils import (DATASET_PATH, GEMMA, OLMO, LLAMA, PROJECT_NAME, RESPONSE_TEMP,
                   SEED, TASK_PREFIX)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def formatting_func(example: DatasetDict) -> List[str]:
    """
    Formats input examples by concatenating the source text with the target text,
    using the task-specific prefix and response template.

    Args:
        example: A dataset dictionary containing 'source' and 'target' fields.

    Returns:
        A list of formatted strings ready for model training.
    """
    output_texts = []
    for i in range(len(example["source"])):
        text = (
            TASK_PREFIX
            + f"{example['source'][i]}{RESPONSE_TEMP} {example['target'][i]}"
        )
        output_texts.append(text)

    return output_texts


if __name__ == "__main__":

    set_seed(SEED + 21)
    parser = argparse.ArgumentParser(description="Supervise Fine-tuning with Gemma-2B, "
                                                 "OLMo-1B, or Llama3-8B.")
    parser.add_argument("--model", type=str,
                        help="Either gemma, olmo, or llama")
    args = parser.parse_args()

    if args.model == "gemma":
        model_name = GEMMA
    elif args.model == "olmo":
        model_name = OLMO
    elif args.model == "llama":
        model_name = LLAMA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        raise ValueError(f"Invalid model name: {args.model}")
    run_name = f'sft_{model_name.split("/")[-1]}'
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    dataset = load_from_disk(DATASET_PATH)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    if model_name == LLAMA:
        model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=f"ckpts/{run_name}",
        overwrite_output_dir=False,
        num_train_epochs=5.0,
        do_train=True,
        do_eval=True,
        do_predict=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
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
        peft_config=lora_config if model_name == LLAMA else None
    )
    wandb.init(project=PROJECT_NAME, name=run_name, config=training_args)

    trainer = SFTTrainer(
        model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        formatting_func=formatting_func,
        max_seq_length=1024,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
