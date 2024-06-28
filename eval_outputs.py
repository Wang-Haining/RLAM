"""
This module implements evaluation functions for sft and policy models.
It uses the same generation config as used in policy rolling out.
A detailed csv as well as an overview of the results will be saved.
"""

import argparse
import csv
import heapq
import json
import os
import pickle
from typing import Dict, List

import evaluate
import numpy as np
import torch
from nltk.tokenize import sent_tokenize
from sacrebleu.metrics import BLEU
from sacremoses import MosesTokenizer
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, GenerationConfig,
                          AutoTokenizer)
from trl import set_seed

from utils import (OLMO_1B, GEMMA_2B, GEMMA_7B, PHI2_3B, LLAMA3_8B, GPT2_XL,
                   SEED, TASK_PREFIX, VOA1500, WORD_ACCESSIBILITY_MODEL, WORD_FREQ_CSV,
                   build_ppo_dataset, compute_ari, compute_flesch_kincaid,
                   compute_sent_len, compute_token_accessibility,
                   read_token_frequencies, MAX_OUTPUT_LENGTHS)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metric_bleu = BLEU()
metric_sari = evaluate.load("sari")
metric_rouge = evaluate.load("rouge")
metric_bertscore = evaluate.load("bertscore")
# get word frequencies and the model to predict relative rare word's accessibility
token_freq = read_token_frequencies(WORD_FREQ_CSV)
top_100k_tokens = heapq.nlargest(100000, token_freq, key=token_freq.get)
# load for making predictions word accessibility
wa_model = pickle.load(open(WORD_ACCESSIBILITY_MODEL, 'rb'))
total_tokens = sum(token_freq.values())
mt = MosesTokenizer(lang='en')
# VOA Word Book, Section A-Z, Science programs, and Organs of the body (1517 in total)
# from https://simple.wikipedia.org/wiki/Wikipedia:VOA_Special_English_Word_Book
# scraped on May 15, 2024
voa1500 = json.load(open(VOA1500, 'r', encoding='utf-8'))


def calculate_metrics(generated_text: str,
                      target_text: str,
                      source_text: str) -> Dict[str, float]:
    metrics_dict = {}
    generated_texts = [generated_text.strip()]
    source_texts = [source_text.strip()]
    target_texts = [[target_text.strip()]]
    metrics_dict.update({"ari": compute_ari(generated_texts[0])})
    metrics_dict.update({"fk": compute_flesch_kincaid(generated_texts[0])})
    metrics_dict.update({"bleu": metric_bleu.corpus_score(generated_texts,
                                                          target_texts).score})
    metrics_dict.update(metric_sari.compute(sources=source_texts,
                                            predictions=generated_texts,
                                            references=target_texts))
    _rouge = metric_rouge.compute(predictions=generated_texts,
                                  references=target_texts)
    metrics_dict.update({"rougeL": _rouge["rougeL"]})
    bertscore_result = metric_bertscore.compute(predictions=generated_texts,
                                                references=target_texts,
                                                lang="en", device="cpu",
                                                model_type='bert-large-uncased')
    metrics_dict.update({"bertscore": np.mean(bertscore_result["f1"])})
    # complexity measure
    word_accessibility_list = []
    sent_len_list = []
    num_words = 0
    num_chars = 0
    num_voa_words = 0
    sents = sent_tokenize(generated_text)
    for sent in sents:
        sent_len_list.append(compute_sent_len(sent))
        for token in mt.tokenize(sent):
            num_words += 1
            num_chars += len(token)
            if token.lower() in voa1500:
                num_voa_words += 1
            word_accessibility_list.append(compute_token_accessibility(token,
                                                                       top_100k_tokens,
                                                                       wa_model,
                                                                       total_tokens,
                                                                       token_freq))
    p = num_voa_words / num_words
    metrics_dict.update({"voa_log_ratio": np.log(p / (1 - p))})
    metrics_dict.update({"avg_sent_len": np.mean(sent_len_list)})
    metrics_dict.update({"avg_word_accessibility": np.mean(word_accessibility_list)})
    metrics_dict.update({'num_sents': len(sents)})
    metrics_dict.update({'avg_word_len': num_chars/num_words})
    return metrics_dict


def evaluate_model(model, dataset, tokenizer, generation_config) -> List[Dict]:
    results = []
    model.eval()
    with torch.no_grad():
        for i, sample in tqdm(enumerate(dataset)):
            input_ids = torch.tensor(sample['query_token']).unsqueeze(0).to(device)
            response_token_ids = model.generate(input_ids=input_ids,
                                                generation_config=generation_config)
            gen_tokens = response_token_ids[0].squeeze()[input_ids.size(1):]
            generated_text = tokenizer.decode(gen_tokens,
                                              skip_special_tokens=True,
                                              clean_up_tokenization_spaces=True).strip()
            result = calculate_metrics(generated_text,
                                       sample['response'],
                                       sample['source'])  # the original abstract
            results.append(result | {'generated_text': generated_text})
    return results


if __name__ == "__main__":
    print('*' * 90)
    parser = argparse.ArgumentParser(
        description="Evaluate SFT and policy model outputs given model type and validation ARI.")
    parser.add_argument("--model", type=str, help="The model type (across runs) to evaluate")
    parser.add_argument("--upper_ari_bound", type=float, default=15.0, help="The upper bound of evaluation ARI for a checkpoint to be considered in the evaluation")
    parser.add_argument("--lower_ari_bound", type=float, default=10.0, help="The lower bound of evaluation ARI for a checkpoint to be considered in the evaluation")
    parser.add_argument("--reward", type=str, default='uam', choices=['uam', 'ari'], help="Reward type")
    args = parser.parse_args()
    set_seed(SEED)
    SAVE_DIR = "eval_results"

    print(f'Starting evaluation: only newly added runs whose checkpoints met '
          f'{args.lower_ari_bound} <= validation ARI <= {args.upper_ari_bound} '
          f'will be evaluated')

    # identify the base model based on the provided model type argument
    if "gemma-2b" in args.model.lower():
        base_model = GEMMA_2B
    elif "gemma-7b" in args.model.lower():
        base_model = GEMMA_7B
    elif "olmo-1b" in args.model.lower():
        base_model = OLMO_1B
    elif 'llama' in args.model.lower():
        base_model = LLAMA3_8B
    elif 'gpt2-xl' in args.model.lower():
        base_model = GPT2_XL
    elif 'phi2-3b' in args.model.lower():
        base_model = PHI2_3B
    else:
        raise ValueError(f"Unknown ckpt path {args.ckpt_path}")

    # define the generation configuration
    test_generation_config = GenerationConfig(
        max_new_tokens=MAX_OUTPUT_LENGTHS[args.model],
        temperature=(0.01 + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
        return_dict_in_generate=True,
        num_return_sequences=1
    )

    dataset = build_ppo_dataset(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # load the overview file if it exists
    overview_path = os.path.join(SAVE_DIR, "overview.jsonl")
    if os.path.exists(overview_path):
        with open(overview_path, mode='r', encoding='utf-8') as f:
            overview = [json.loads(line) for line in f]
    else:
        overview = []
    evaluated_runs = {entry["ckpt_path"] for entry in overview}

    # check and evaluate SFT models
    # SFT runs have slightly different naming conventions
    sft_base_model = args.model.split("/")[-1]
    sft_model_dir = os.path.join("ckpts", f"sft_{sft_base_model}")
    if os.path.exists(sft_model_dir) and sft_model_dir not in evaluated_runs:
        sft_checkpoints = os.listdir(sft_model_dir)
        if len(sft_checkpoints) != 1:
            raise ValueError(
                f"Expected exactly one checkpoint in {sft_model_dir}, but found {len(sft_checkpoints)}.")
        sft_checkpoint = sft_checkpoints[0]
        sft_ckpt_path = os.path.join(sft_model_dir, sft_checkpoint)
        print(f'Starting evaluation for {sft_ckpt_path}')
        model = AutoModelForCausalLM.from_pretrained(sft_ckpt_path,
                                                     torch_dtype=torch.bfloat16)
        model.to(device)

        # evaluate with test generation config
        eval_results = evaluate_model(model, dataset["test"], tokenizer,
                                      test_generation_config)

        # save evaluation results to CSV
        file_path = os.path.join(SAVE_DIR, f"sft_{sft_base_model}___{sft_checkpoint}.csv")
        with open(file_path, mode="w", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=eval_results[0].keys())
            writer.writeheader()
            writer.writerows(eval_results)

        # calculate average and standard deviation of scores
        avg_scores = {f"avg_{metric}": np.mean([x[metric] for x in eval_results]) for
                      metric in eval_results[0].keys() if
                      metric not in ["generated_text"]}
        std_scores = {f"std_{metric}": np.std([x[metric] for x in eval_results]) for
                      metric in eval_results[0].keys() if
                      metric not in ["generated_text"]}

        # save the overview in JSONL format
        with open(overview_path, mode='a', encoding='utf-8') as f:
            json.dump({"ckpt_path": sft_ckpt_path} | avg_scores | std_scores, f)
            f.write('\n')

        # print out results
        print('*' * 90)
        print(f'SFT performance for {sft_base_model}:')
        print("Average scores for {}: {}".format(sft_ckpt_path, avg_scores))
        print(
            "Standard deviation of scores for {}: {}".format(sft_ckpt_path, std_scores))
        print('*' * 90)

    # get the relevant PPO runs using heuristics
    relevant_runs = []
    for run in os.listdir("ckpts"):
        if run.startswith(f"ppo_{args.reward}_{args.model}"):
            if run not in evaluated_runs:
                relevant_runs.append(run)
                print(f'{len(relevant_runs)} runs will be evaluated: {evaluated_runs}')

    for run in relevant_runs:
        ckpt_dir = os.path.join("ckpts", run)
        for ckpt in os.listdir(ckpt_dir):
            if ckpt.startswith("step_") and ckpt.endswith(
                    "ari_{}.pt".format(args.upper_ari_bound)):
                ari = float(ckpt.split("_ari_")[1])
                if args.lower_ari_bound <= ari <= args.upper_ari_bound:
                    ckpt_path = os.path.join(ckpt_dir, ckpt)
                    print(f'Starting evaluation for {ckpt_path}')
                    model = AutoModelForCausalLM.from_pretrained(ckpt_path,
                                                                 torch_dtype=torch.bfloat16)
                    model.to(device)

                    # evaluate with test generation config
                    eval_results = evaluate_model(model, dataset["test"], tokenizer,
                                                  test_generation_config)

                    # save evaluation results to CSV
                    file_path = os.path.join(SAVE_DIR, f"{run}___{ckpt}.csv")
                    with open(file_path, mode="w", encoding="utf-8") as file:
                        writer = csv.DictWriter(file, fieldnames=eval_results[0].keys())
                        writer.writeheader()
                        writer.writerows(eval_results)

                    # calculate average and standard deviation of scores
                    avg_scores = {
                        f"avg_{metric}": np.mean([x[metric] for x in eval_results]) for
                        metric in eval_results[0].keys() if
                        metric not in ["generated_text"]}
                    std_scores = {
                        f"std_{metric}": np.std([x[metric] for x in eval_results]) for
                        metric in eval_results[0].keys() if
                        metric not in ["generated_text"]}

                    # save the overview in JSONL format
                    with open(overview_path, mode='a', encoding='utf-8') as f:
                        json.dump({"ckpt_path": ckpt_path} | avg_scores | std_scores, f)
                        f.write('\n')

                    # print out results
                    print('*' * 90)
                    print(f'RLUAM performance for {run} {ckpt}:')
                    print("Average scores for {}: {}".format(ckpt_path, avg_scores))
                    print("Standard deviation of scores for {}: {}".format(ckpt_path,
                                                                           std_scores))
                    print('*' * 90)
    print('*' * 90)
