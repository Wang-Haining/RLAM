import os

import torch
import wandb
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from utils import DATASET_PATH, SEED

os.environ["TOKENIZERS_PARALLELISM"] = "false"
MODEL_NAME = "google/gemma-2b"
RESPONSE_TEMP = "\n### Answer:"
project_name = f'sft_{MODEL_NAME.split("/")[-1]}'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
collator = DataCollatorForCompletionOnlyLM(RESPONSE_TEMP, tokenizer=tokenizer)


def formatting_func(example):
    output_texts = []
    for i in range(len(example["source"])):
        text = f"### Simplify the scholarly abstract so it is immediately understandable to a layperson: {example['source'][i]}{RESPONSE_TEMP} {example['target'][i]}"
        output_texts.append(text)

    return output_texts


if __name__ == "__main__":

    torch.manual_seed(SEED)

    dataset = load_from_disk(DATASET_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=f"ckpts/{project_name}",
        overwrite_output_dir=False,
        num_train_epochs=10,
        do_train=True,
        do_eval=True,
        do_predict=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=3e-5,
        weight_decay=1e-1,
        logging_steps=20,
        eval_steps=40,
        bf16=True,
        report_to="wandb",
        load_best_model_at_end=True,
        save_steps=200,
        save_total_limit=3,
        remove_unused_columns=True,
    )
    wandb.init(project=project_name, config=training_args)

    trainer = SFTTrainer(
        model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        formatting_func=formatting_func,
        data_collator=collator,
        max_seq_length=1024,
        args=training_args,
    )

    trainer.train()
