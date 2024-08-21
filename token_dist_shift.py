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
    output_csv: str = None,
    verbose: bool = True
):
    """
    analyze the token distribution shift between sft and ppo models using a single
    forward pass, and store results in a csv

    args:
        sft_model: the sft model
        tokenizer: the tokenizer
        query: the input query text
        ppo_text: the generated text from the ppo model
        output_csv: path to the output csv file
        verbose: whether to print the results during the analysis

    returns:
        a tuple of two lists containing token id, sft rank, ppo rank, shift category,
        and decoded texts
    """
    # encode the query and ppo_text
    query_tokens = tokenizer(query, return_tensors="pt")["input_ids"]
    context_tokens = tokenizer(query + ppo_text, return_tensors="pt")["input_ids"]

    # extract ppo_tokens by subtracting query_tokens length from context_tokens
    ppo_tokens = context_tokens[:, query_tokens.shape[1]:].squeeze()

    # prepare a mask for each position
    attention_mask = torch.ones_like(context_tokens)

    token_shifts = []
    decoded_sft_text = []

    with torch.no_grad():
        # get logits for all positions in one forward pass
        sft_logits = sft_model(context_tokens.to(sft_model.device),
                               attention_mask=attention_mask.to(
                                   sft_model.device)).logits

        # decode the greedy output from the sft logits
        sft_generated_ids = sft_logits.argmax(dim=-1)
        sft_generated_text = tokenizer.decode(sft_generated_ids[0, query_tokens.shape[1]:],
                                              skip_special_tokens=True)
        decoded_sft_text.append(sft_generated_text)

    # iterate over each token in ppo_tokens
    for t in range(ppo_tokens.shape[0]):
        # compute the softmax probabilities for the generated token positions
        sft_probs = torch.softmax(sft_logits[0, query_tokens.shape[1] + t - 1, :],
                                  dim=-1).cpu().numpy().flatten()

        ppo_token_id = ppo_tokens[t].item()

        # get the rank of the ppo_token in the sft model's prediction
        sft_rank = np.argsort(-sft_probs).tolist().index(ppo_token_id)

        # get top 5 most likely tokens predicted by the sft model
        top_tokens = np.argsort(-sft_probs)[:10]
        top_tokens_decoded = tokenizer.decode(top_tokens).split()

        # debugging prints to check alignment and correctness
        if verbose:
            print(f"processing token at position {query_tokens.shape[1] + t}:")
            print(f"top 5 predicted tokens: {top_tokens_decoded}")
            print(f"token: {tokenizer.decode([ppo_token_id])}, sft rank: {sft_rank}")

        shift_category = (
            "unshifted"
            if sft_rank == 0
            else "marginal" if 1 <= sft_rank <= 2 else "shifted"
        )

        token_shifts.append({
            "token_id": ppo_token_id,
            "sft_rank": sft_rank,
            "shift_category": shift_category,
            "token": tokenizer.decode([ppo_token_id]),
            "top_predicted_tokens": top_tokens_decoded
        })

    # save results to csv
    df = pd.DataFrame(token_shifts)
    df["ppo_text"] = ppo_text
    df["sft_generated_text"] = decoded_sft_text[0]
    df.to_csv(output_csv, index=False)

    return token_shifts, sft_generated_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Token Distribution Shift Analysis "
                                                 "with RLAM Model Generations")
    parser.add_argument("--sft_model_path", type=str,
                        default="ckpts/sft_gemma-2b/checkpoint-1680",
                        help="The path to the sft model")
    parser.add_argument("--csv_file", type=str, required=True,
                        help="CSV file containing RLAM model generations")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Output CSV file")
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
            sft_model, tokenizer, query, ppo_text,
            output_csv=args.output_csv,
        )

        # print the results for each query
        print(f"Query:\n{query}")
        print(f"Generated Text:\n{generated_text}\n")
        for token_shift in token_shifts:
            token = token_shift["token"]
            sft_rank = token_shift["sft_rank"]
            shift_category = token_shift["shift_category"]
            print(
                f"Token: {token}, SFT Rank: {sft_rank}, Category: {shift_category}"
            )
        print("\n" + "="*50 + "\n")
