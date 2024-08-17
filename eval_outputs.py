"""
This module implements evaluation functions for sft and policy models.
it uses the same generation config as used in policy rolling out.
a detailed csv as well as an overview of the results will be saved.
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
from transformers import (AutoModelForCausalLM, AutoTokenizer, GenerationConfig)

from utils import (GEMMA_2B, GEMMA_7B, MAX_OUTPUT_LENGTHS, OLMO_1B, PHI2_3B,
                   SEED, VOA1500, WORD_ACCESSIBILITY_MODEL, WORD_FREQ_CSV,
                   build_sass_dataset, compute_ari, compute_flesch_kincaid,
                   compute_sent_len, compute_token_accessibility,
                   read_token_frequencies)

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
wa_model = pickle.load(open(WORD_ACCESSIBILITY_MODEL, "rb"))
total_tokens = sum(token_freq.values())
mt = MosesTokenizer(lang="en")
# voa word book, section a-z, science programs, and organs of the body (1517 in total)
# from https://simple.wikipedia.org/wiki/wikipedia:voa_special_english_word_book
# scraped on may 15, 2024
voa1500 = json.load(open(VOA1500, "r", encoding="utf-8"))


def calculate_metrics(
        generated_text: str, target_text: str, source_text: str
) -> Dict[str, float]:
    metrics_dict = {}
    generated_texts = [generated_text.strip()]
    source_texts = [source_text.strip()]
    target_texts = [[target_text.strip()]]
    metrics_dict.update({"ari": compute_ari(generated_texts[0])})
    metrics_dict.update({"fk": compute_flesch_kincaid(generated_texts[0])})
    metrics_dict.update(
        {"bleu": metric_bleu.corpus_score(generated_texts, target_texts).score}
    )
    metrics_dict.update(
        metric_sari.compute(
            sources=source_texts, predictions=generated_texts, references=target_texts
        )
    )
    _rouge = metric_rouge.compute(predictions=generated_texts, references=target_texts)
    metrics_dict.update({"rougeL": _rouge["rougeL"]})
    bertscore_result = metric_bertscore.compute(
        predictions=generated_texts,
        references=target_texts,
        lang="en",
        device="cpu",
        model_type="bert-large-uncased",
    )
    metrics_dict.update({"bertscore": np.mean(bertscore_result["f1"])})
    # complexity measure
    word_accessibility_list = []
    avg_sent_word_accessibility_lists = []
    sent_len_list = []
    num_words = 0
    num_chars = 0
    num_voa_words = 0
    sents = sent_tokenize(generated_text)
    for sent in sents:
        sent_word_accessibility_list = []
        sent_len_list.append(compute_sent_len(sent))
        for token in mt.tokenize(sent, escape=False):
            num_words += 1
            num_chars += len(token)
            if token.lower() in voa1500:
                num_voa_words += 1
            sent_word_accessibility_list.append(
                compute_token_accessibility(
                    token, top_100k_tokens, wa_model, total_tokens, token_freq
                )
            )
        avg_sent_word_accessibility_lists.append(np.mean(sent_word_accessibility_list))
        word_accessibility_list.extend(sent_word_accessibility_list)
    p = (num_voa_words / num_words) + 1e-12
    metrics_dict.update({"voa_log_ratio": np.log(p / (1 - p))})
    metrics_dict.update({"avg_sent_len": np.mean(sent_len_list)})
    metrics_dict.update({"avg_word_accessibility": np.mean(word_accessibility_list)})
    metrics_dict.update({"num_sents": len(sents)})
    metrics_dict.update({"avg_word_len": num_chars / num_words})
    metrics_dict.update({"sent_word_accessibility_std": np.std(avg_sent_word_accessibility_lists)})
    return metrics_dict


def evaluate_model(
        model, dataset, tokenizer, generation_config, batch_size, model_type='clm',
        verbose=False
) -> List[Dict]:
    results = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch_samples = dataset[i: i + batch_size]
            # it is good to retokenize the ['query'] column for batch processing
            input_ids = torch.tensor(batch_samples["query_token"]).to(device)
            generated_tokens = model.generate(
                input_ids=input_ids, generation_config=generation_config
            )
            # only newly generated text are returned
            if model_type == 'clm':
                generated_texts = tokenizer.batch_decode(
                    generated_tokens[:, input_ids.shape[1]:],
                    skip_special_tokens=True,
                )
            elif model_type == 'seq2seq':
                generated_texts = tokenizer.batch_decode(generated_tokens,
                                                         skip_special_tokens=True)
            for j, generated_text in enumerate(generated_texts):
                generated_text = generated_text.strip()
                result = calculate_metrics(
                    generated_text,
                    batch_samples["response"][j],
                    batch_samples["source"][j],
                )
                if verbose:
                    print(f'{generated_text=}')
                results.append(result | {"generated_text": generated_text})
    return results


if __name__ == "__main__":
    print("*" * 90)
    parser = argparse.ArgumentParser(
        description="evaluate sft and policy model outputs for multiple checkpoints"
    )
    parser.add_argument(
        "--sft_ckpt", type=str, help="To init a proper tokenizer")
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="path containing folders of specific checkpoints to evaluate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=20, help="batch size for inference"
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0, help="sampling top_p"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.01, help="sampling temperature"
    )
    parser.add_argument(
        "--upper_ari_bound",
        type=float,
        default=15.0,
        help="the upper bound of evaluation ari for a checkpoint to be considered in the evaluation",
    )
    parser.add_argument(
        "--lower_ari_bound",
        type=float,
        default=8.0,
        help="the lower bound of evaluation ari for a checkpoint to be considered in the evaluation",
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="flag to print generated texts during evaluation. defaults to false.",
    )
    args = parser.parse_args()
    torch.manual_seed(SEED)
    save_dir = f"eval_results_temp_{args.temperature}"
    os.makedirs(save_dir, exist_ok=True)

    # load the overview file if it exists
    overview_path = os.path.join(save_dir, "overview.jsonl")
    if os.path.exists(overview_path):
        with open(overview_path, mode="r", encoding="utf-8") as f:
            overview = [json.loads(line) for line in f]
    else:
        overview = []
    evaluated_runs = {entry["run_path"] for entry in overview}

    # iterate through each checkpoint folder
    checkpoint_dirs = [os.path.join(args.ckpt_path, d) for d in os.listdir(args.ckpt_path) if os.path.isdir(os.path.join(args.ckpt_path, d))]

    for checkpoint_dir in checkpoint_dirs:
        print(f"evaluating checkpoint in directory: {checkpoint_dir}")
        base_model = os.path.basename(checkpoint_dir).split('_')[0]

        # iterate over each checkpoint within the folder
        # if checkpoint_dir.split('/')[-1].startswith("step_"):
        ari = float(checkpoint_dir.split("_ari_")[-1])
        print(f'Eval ari is {ari}')
        if args.lower_ari_bound <= ari <= args.upper_ari_bound:
            # load the corresponding tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(args.sft_ckpt)
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_dir, torch_dtype=torch.bfloat16
            )
            model.to(device)

            # define the generation configuration
            test_generation_config = GenerationConfig(
                max_new_tokens=MAX_OUTPUT_LENGTHS[base_model.lower()],
                temperature=args.temperature + 1e-7,
                top_k=0.0,
                top_p=args.top_p,
                do_sample=True,
                num_return_sequences=1,
            )
            print(f"{test_generation_config=}")

            # load dataset
            dataset = build_sass_dataset(args.sft_ckpt, base_model, 'left')

            # evaluate the model
            eval_results = evaluate_model(
                model,
                dataset["test"],
                tokenizer,
                test_generation_config,
                batch_size=args.batch_size,
                model_type='clm',
                verbose=args.verbose
            )

            # save evaluation results to csv
            file_path = os.path.join(save_dir, f"{checkpoint_dir.replace('/', '|')}.csv")
            with open(file_path, mode="w", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=eval_results[0].keys())
                writer.writeheader()
                writer.writerows(eval_results)

            # calculate average and standard deviation of scores
            avg_scores = {
                f"avg_{metric}": np.mean([x[metric] for x in eval_results])
                for metric in eval_results[0].keys()
                if metric not in ["generated_text"]
            }
            std_scores = {
                f"std_{metric}": np.std([x[metric] for x in eval_results])
                for metric in eval_results[0].keys()
                if metric not in ["generated_text"]
            }

            # save the overview in jsonl format
            with open(overview_path, mode="a", encoding="utf-8") as f:
                json.dump(
                    {"run_path": checkpoint_dir}
                    | {"ckpt_path": checkpoint_dir}
                    | avg_scores
                    | std_scores,
                    f,
                )
                f.write("\n")

            # print out results
            print("*" * 90)
            print(f"performance for {checkpoint_dir} at temperature {args.temperature}:")
            print(f"average scores: {avg_scores}")
            print(f"standard deviation of scores: {std_scores}")
            print("*" * 90)
