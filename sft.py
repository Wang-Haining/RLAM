import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from utils import DATASET_PATH, SEED

MODEL_NAME = 'meta-llama/Llama-2-7b-hf'
RESPONSE_TEMP = "### Answer:"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
collator = DataCollatorForCompletionOnlyLM(RESPONSE_TEMP, tokenizer=tokenizer)


# def formatting_func(example):
#     output_texts = []
#     for i in range(len(example['instruction'])):
#         text = (
#             f"### Please simplify the scholarly abstract so it is immediately "
#             f"understandable to a layperson: {example['source'][i]}\n {
#             RESPONSE_TEMP} "
#             f"{example['target'][i]}")
#         output_texts.append(text)
#     return output_texts


def formatting_func(example):
    text = (f"### Please simplify the scholarly abstract so it is immediately "
            f"understandable to a layperson: {example['source']}\n {RESPONS
            E_TEMP} {example['target']}")
    return text


if __name__ == "__main__":
    torch.manual_seed(SEED)

    dataset = load_from_disk(DATASET_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)

    common_args = {'output_dir': 'ckpts/sft_llama2-7b-hf',
                   'overwrite_output_dir': False,
                   'do_train': True,
                   'do_eval': True,
                   'do_predict': True,
                   'evaluation_strategy': 'steps',
                   'per_device_train_batch_size': 4,
                   'per_device_eval_batch_size': 4,
                   'gradient_accumulation_steps': 4,
                   'learning_rate': 3e-5,
                   'weight_decay': 1e-1,
                   'logging_steps': 20,
                   'eval_steps': 200,
                   'bf16': True,
                   'run_name': 'sft_llama2-7b',
                   'report_to': 'wandb',
                   'load_best_model_at_end': True,
                   'save_steps': 200,
                   'save_total_limit': 3,
                   'remove_unused_columns': True
                   }

    trainer = SFTTrainer(
        model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        formatting_func=formatting_func,
        data_collator=collator,
        max_seq_length=1024,
        **common_args
    )

    trainer.train()
