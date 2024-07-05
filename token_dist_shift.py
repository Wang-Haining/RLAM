"""
This module implements token distribution shift analysis.

References:
    - https://allenai.github.io/re-align/tds.html
"""

from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import MAX_OUTPUT_LENGTHS, OLMO_1B, build_ppo_dataset


def load_models_and_tokenizer(
    base_model: str, sft_ckpt: str, ppo_ckpt: str, device: str
) -> Tuple:
    """
    Load the tokenizer and models for base, SFT, and PPO with bf16 precision on cuda.

    Args:
        base_model: The base model identifier.
        sft_ckpt: Checkpoint for the SFT model.
        ppo_ckpt: Checkpoint for the PPO model.
        device: Cuda or cpu.

    Returns:
        tuple: Tokenizer, SFT model, and PPO model.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    sft_model = AutoModelForCausalLM.from_pretrained(
        sft_ckpt, torch_dtype=torch.bfloat16
    ).to(device)
    ppo_model = AutoModelForCausalLM.from_pretrained(
        ppo_ckpt, torch_dtype=torch.bfloat16
    ).to(device)
    return tokenizer, sft_model, ppo_model


def generate_output(
    model: AutoModelForCausalLM,
    base_model: str,
    input_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
):
    """
    Generate output using the model with greedy decoding.

    Args:
        model: The language model.
        base_model: The base model identifier.
        input_ids: The input IDs.
        tokenizer: The tokenizer.

    Returns:
        tuple: The generated output IDs and decoded text.
    """
    try:
        output_len = MAX_OUTPUT_LENGTHS[base_model.split("/")[-1]]
    except KeyError:
        raise KeyError(f"Illegal {base_model}.")
    with torch.no_grad():
        output = model.generate(
            input_ids.to(model.device),
            max_length=input_ids.shape[1] + output_len,
            do_sample=False,
        )
    return output, tokenizer.decode(output[0], skip_special_tokens=True)


def analyze_token_distribution_shift(
    base_model: str,
    sft_model: AutoModelForCausalLM,
    ppo_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    query: str,
):
    """
    Analyze the token distribution shift between SFT and PPO models.

    Args:
        base_model: The base model identifier.
        sft_model: The SFT model.
        ppo_model: The PPO model.
        tokenizer: The tokenizer.
        query: The input query text.

    Returns:
        list: A list of tuples containing token ID, SFT rank, PPO rank, and shift category.
    """
    input_ids = tokenizer(query, return_tensors="pt")["input_ids"]
    query_length = input_ids.shape[1]

    ppo_output, ppo_text = generate_output(ppo_model, base_model, input_ids, tokenizer)
    ppo_tokens = tokenizer(ppo_text, return_tensors="pt")["input_ids"][0]

    token_shifts = []
    for t in tqdm(range(query_length, len(ppo_tokens))):
        context_tokens = ppo_tokens[:t]
        context_text = tokenizer.decode(context_tokens, skip_special_tokens=True)
        context_ids_sft = tokenizer(context_text, return_tensors="pt")["input_ids"]

        with torch.no_grad():
            sft_logits = sft_model(context_ids_sft.to(sft_model.device)).logits
            ppo_logits = ppo_model(ppo_tokens.unsqueeze(0).to(ppo_model.device)).logits

        sft_probs = torch.softmax(sft_logits[:, -1, :], dim=-1).cpu().numpy().flatten()
        ppo_probs = (
            torch.softmax(ppo_logits[:, t - 1, :], dim=-1).cpu().numpy().flatten()
        )

        ppo_token_id = ppo_tokens[t].item()

        sft_rank = np.argsort(-sft_probs).tolist().index(ppo_token_id)
        ppo_rank = np.argsort(-ppo_probs).tolist().index(ppo_token_id)

        shift_category = (
            "unshifted"
            if sft_rank == 0
            else "marginal" if 1 <= sft_rank <= 2 else "shifted"
        )

        token_shifts.append((ppo_token_id, sft_rank, ppo_rank, shift_category))

    return token_shifts, ppo_text


# if __name__ == '__main__':
#
#     set_seed(SEED + 2122)
#     parser = argparse.ArgumentParser(description="Supervise Fine-tuning with "
#                                                  "Gemma-2B/7B, OLMo-1B, Llama3-8B or Phi-2.")
#     parser.add_argument("--model", type=str,
#                         choices=["gemma-2b", "gemma-7b", "olmo-1b", "llama3-8b", "gpt2-xl", 'phi2-3b'],
#                         help="Either gemma-2b, gemma-7b, olmo-1b, llama3-8b, gpt2-xl, or phi2-3b")
#     parser.add_argument("--learning_rate", type=float, default=1e-5)
#     parser.add_argument("--per_device_train_batch_size", type=int, default=2)
#     parser.add_argument("--is_peft_model",
#                         type=lambda x: (str(x).lower() == 'true'), default=False,
#                         help="Whether to use LoRA for finetuning")
#     args = parser.parse_args()

# Define the model paths and dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
base_model = OLMO_1B
sft_ckpt = "ckpts/sft_OLMo-1B-hf/checkpoint-1100"
ppo_ckpt = "ckpts/ppo_uam_olmo-1b_dynamic_kl_control__42__1719962332/step_220_ari_12.06"
dataset = build_ppo_dataset(OLMO_1B)["test"]

# extract a query from the dataset
query = dataset["query"][0]

# load the tokenizer and models
tokenizer, sft_model, ppo_model = load_models_and_tokenizer(
    base_model, sft_ckpt, ppo_ckpt, device
)

# analyze the token distribution shift for the given query
token_shifts, ppo_text = analyze_token_distribution_shift(
    base_model, sft_model, ppo_model, tokenizer, query
)

# print the whole generated text first
print(f"Generated Text: {ppo_text}\n")

# print the results
for token_id, sft_rank, ppo_rank, shift_category in token_shifts:
    token = tokenizer.decode([token_id])
    print(
        f"Token: {token}, SFT Rank: {sft_rank}, PPO Rank: {ppo_rank}, Category: {shift_category}"
    )
