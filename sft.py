"""
This module performs supervised finetuning on the OLMo, Gemma, and LLama3 using the
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
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer,
                          EarlyStoppingCallback, TrainingArguments)
from trl import SFTTrainer, set_seed
from peft import LoraConfig, get_peft_model

from utils import (DATASET_PATH, OLMO_1B, GEMMA_2B, GEMMA_7B, LLAMA3_8B, GPT2_XL,
                   PROJECT_NAME, RESPONSE_TEMP, SEED, TASK_PREFIX, CKPTS_DIR)

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

    set_seed(SEED + 2122)
    parser = argparse.ArgumentParser(description="Supervise Fine-tuning with "
                                                 "Gemma-2B/7B, OLMo-1B, or Llama3-8B.")
    parser.add_argument("--model", type=str,
                        choices=["gemma-2b", "gemma-7b", "olmo-1b", "llama3-8b", "gpt2-xl"],
                        help="Either gemma-2b, gemma-7b, olmo-1b, llama3-8b, or gpt2-xl")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--is_peft_model",
                        type=lambda x: (str(x).lower() == 'true'), default=False,
                        help="Whether to use LoRA for finetuning")
    args = parser.parse_args()

    if args.model == "gemma-2b":
        model_name = GEMMA_2B
    elif args.model == "olmo-1b":
        model_name = OLMO_1B
    elif args.model == "gemma-7b":
        model_name = GEMMA_7B
    elif args.model == "llama3-8b":
        model_name = LLAMA3_8B
    elif args.model == "gpt2-xl":
        model_name = GPT2_XL
    else:
        raise ValueError(f"Invalid model name: {args.model}")

    # lora config if necessary
    if args.is_peft_model:
        lora_config = LoraConfig(
            init_lora_weights="gaussian",
            target_modules=["q_proj", "v_proj"],
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    run_name = f'sft_{model_name.split("/")[-1]}'
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    dataset = load_from_disk(DATASET_PATH)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.bfloat16)

    if args.is_peft_model:
        model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=f"{CKPTS_DIR}/{run_name}",
        overwrite_output_dir=False,
        num_train_epochs=5.0,
        do_train=True,
        do_eval=True,
        do_predict=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,  # same to training
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

    trainer = SFTTrainer(
        model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        formatting_func=formatting_func,
        max_seq_length=768,
        args=training_args,
        peft_config=lora_config if args.is_peft_model else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
