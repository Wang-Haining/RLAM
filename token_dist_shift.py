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

#
# def analyze_token_distribution_shift(
#     sft_model: AutoModelForCausalLM,
#     tokenizer: AutoTokenizer,
#     query: str,
#     ppo_text: str,
#     verbose: bool = True
# ):
#     """
#     Analyze the token distribution shift between SFT and PPO models.
#
#     Args:
#         sft_model: The SFT model.
#         tokenizer: The tokenizer.
#         query: The input query text.
#         ppo_text: The generated text from the PPO model.
#         verbose: Whether to print the results during the analysis.
#
#     Returns:
#         A list of tuples containing token ID, SFT rank, PPO rank, and shift category.
#     """
#     input_ids = tokenizer(query, return_tensors="pt")["input_ids"]
#     query_length = input_ids.shape[1]
#
#     ppo_tokens = tokenizer(ppo_text, return_tensors="pt")["input_ids"][0]
#
#     token_shifts = []
#     for t in tqdm(range(1, len(ppo_tokens) + 1)):
#         context_tokens = torch.cat([input_ids, ppo_tokens[:t].unsqueeze(0)], dim=1)
#
#         with torch.no_grad():
#             sft_logits = sft_model(context_tokens.to(sft_model.device)).logits
#
#         sft_probs = torch.softmax(sft_logits[:, -1, :], dim=-1).cpu().numpy().flatten()
#
#         ppo_token_id = ppo_tokens[t - 1].item()
#
#         sft_rank = np.argsort(-sft_probs).tolist().index(ppo_token_id)
#
#         shift_category = (
#             "unshifted"
#             if sft_rank == 0
#             else "marginal" if 1 <= sft_rank <= 2 else "shifted"
#         )
#
#         token_shifts.append((ppo_token_id, sft_rank, shift_category))
#
#         if verbose:
#             token = tokenizer.decode([ppo_token_id])
#             print(
#                 f"Token: {token}, SFT Rank: {sft_rank}, Category: {shift_category}"
#             )
#
#     return token_shifts, ppo_text


def analyze_token_distribution_shift(
    sft_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    query: str,
    ppo_text: str,
    verbose: bool = True
):
    """
    Analyze the token distribution shift between sft and ppo models using a single
    forward pass

    args:
        sft_model: the sft model
        tokenizer: the tokenizer
        query: the input query text
        ppo_text: the generated text from the ppo model
        verbose: whether to print the results during the analysis

    returns:
        a list of tuples containing token id, sft rank, ppo rank, and shift category
    """
    # encode the query and ppo_text
    query_tokens = tokenizer(query, return_tensors="pt")["input_ids"]
    context_tokens = tokenizer(query + ppo_text, return_tensors="pt")["input_ids"]

    # extract ppo_tokens by subtracting query_tokens length from context_tokens
    ppo_tokens = context_tokens[:, query_tokens.shape[1]:].squeeze()

    # prepare a mask for each position
    attention_mask = torch.ones_like(context_tokens)

    token_shifts = []

    with torch.no_grad():
        # get logits for all positions in one forward pass
        sft_logits = sft_model(context_tokens.to(sft_model.device),
                               attention_mask=attention_mask.to(
                                   sft_model.device)).logits

    # iterate over each token in ppo_tokens
    for t in range(ppo_tokens.shape[0]):
        current_position = query_tokens.shape[1] + t

        # compute the softmax probabilities for the generated token positions
        sft_probs = torch.softmax(sft_logits[0, current_position, :], dim=-1).cpu().numpy().flatten()

        ppo_token_id = ppo_tokens[t].item()

        # get the rank of the ppo_token in the sft model's prediction
        sft_rank = np.argsort(-sft_probs).tolist().index(ppo_token_id)

        # get top 5 most likely tokens predicted by the sft model
        top_tokens = np.argsort(-sft_probs)[:5]
        top_tokens_decoded = tokenizer.decode(top_tokens).split()

        # debugging prints to check alignment and correctness
        if verbose:
            print(f"processing token at position {current_position}:")
            print(f"top 5 predicted tokens: {top_tokens_decoded}")
            print(f"token: {tokenizer.decode([ppo_token_id])}, sft rank: {sft_rank}")

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        print(f"Query:\n{query}")
        print(f"Generated Text:\n{generated_text}\n")
        for token_id, sft_rank, shift_category in token_shifts:
            token = tokenizer.decode([token_id])
            print(
                f"Token: {token}, SFT Rank: {sft_rank}, Category: {shift_category}"
            )
        print("\n" + "="*50 + "\n")
