"""
This module implements token distribution shift analysis.

References:
    - https://allenai.github.io/re-align/tds.html
"""


import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import GEMMA_2B, build_sass_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def analyze_token_distribution_shift(
    sft_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    query: str,
    ppo_text: str,
):
    """
    Analyze the token distribution shift between SFT and PPO models.

    Args:
        sft_model: The SFT model.
        tokenizer: The tokenizer.
        query: The input query text.
        ppo_text: The generated text from the PPO model.

    Returns:
        A list of tuples containing token ID, SFT rank, PPO rank, and shift category.
    """
    input_ids = tokenizer(query, return_tensors="pt")["input_ids"]
    query_length = input_ids.shape[1]

    ppo_tokens = tokenizer(ppo_text, return_tensors="pt")["input_ids"][0]

    token_shifts = []
    for t in tqdm(range(query_length, len(ppo_tokens))):
        context_tokens = ppo_tokens[:t]
        context_text = tokenizer.decode(context_tokens, skip_special_tokens=True)
        context_ids_sft = tokenizer(context_text, return_tensors="pt")["input_ids"]

        with torch.no_grad():
            sft_logits = sft_model(context_ids_sft.to(sft_model.device)).logits

        sft_probs = torch.softmax(sft_logits[:, -1, :], dim=-1).cpu().numpy().flatten()

        ppo_token_id = ppo_tokens[t].item()

        sft_rank = np.argsort(-sft_probs).tolist().index(ppo_token_id)

        shift_category = (
            "unshifted"
            if sft_rank == 0
            else "marginal" if 1 <= sft_rank <= 2 else "shifted"
        )

        token_shifts.append((ppo_token_id, sft_rank, shift_category))

    return token_shifts, ppo_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Token Distribution Shift Analysis "
                                                 "with RLAM Model Generations")
    parser.add_argument("--sft_model_path", type=str,
                        default="ckpts/sft_gemma-2b/checkpoint-1680",
                        help="The path to the sft model")
    parser.add_argument("--csv_file", type=str, required=True,
                        help="CSV file containing RLAM model generations")
    args = parser.parse_args()

    # load the dataset and generations from the CSV file
    df = pd.read_csv(args.csv_file)
    test_set = build_sass_dataset(args.sft_model_path, GEMMA_2B)['test']

    # load the sft model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(GEMMA_2B)
    sft_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path, torch_dtype=torch.bfloat16
    ).to(device)

    # iterate over the dataset rows to analyze token distribution shifts
    for index, row in df.iterrows():
        query = test_set['query'][index]
        ppo_text = row['generated_text']

        token_shifts, generated_text = analyze_token_distribution_shift(
            sft_model, tokenizer, query, ppo_text
        )

        # print the results for each query
        print(f"Query: {query}")
        print(f"Generated Text: {generated_text}\n")
        for token_id, sft_rank, shift_category in token_shifts:
            token = tokenizer.decode([token_id])
            print(
                f"Token: {token}, SFT Rank: {sft_rank}, Category: {shift_category}"
            )
        print("\n" + "="*50 + "\n")
