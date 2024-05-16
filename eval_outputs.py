"""
This module implements evaluation functions for sft and policy models. By default, we
evaluate with two generation settings.
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
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          AutoTokenizer)
from trl import set_seed

from utils import (FLAN_T5_TASK_PREFIX, SEED, TASK_PREFIX, VOA1500,
                   WORD_ACCESSIBILITY_MODEL, WORD_FREQ_CSV, build_dataset,
                   compute_ari, compute_sent_len, compute_token_accessibility,
                   read_token_frequencies, compute_flesch_kincaid)

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

# heuristic generation config
heuristic_generation_kwargs = {
    "top_p": .9,
    "max_new_tokens": 300,
    "num_beams": 4,
    "length_penalty": .9,
    "do_sample": True,
    "return_dict_in_generate": True,
    "num_return_sequences": 1,
}

# basic generation config
basic_generation_kwargs = {
    "max_new_tokens": 300,
    "do_sample": True,
    "return_dict_in_generate": True,
    "num_return_sequences": 1,
}


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
                                                lang="en", device="cpu")
    metrics_dict.update({"bertscore": np.mean(bertscore_result["f1"])})
    # complexity measure
    word_accessibility_list = []
    sent_len_list = []
    num_words = 0
    num_voa_words = 0
    sents = sent_tokenize(generated_text)
    for sent in sents:
        sent_len_list.append(compute_sent_len(sent))
        for token in mt.tokenize(sent):
            num_words += 1
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

    # VOA num/proportion

    return metrics_dict


def evaluate_model(model, dataset, tokenizer, generation_kwargs) -> List[Dict]:
    results = []
    model.eval()
    with (torch.no_grad()):
        for i, sample in tqdm(enumerate(dataset)):
            input_ids = sample['input_ids'].unsqueeze(0).to(device)
            response_token_ids = model.generate(input_ids=input_ids,
                                                **generation_kwargs)
            gen_tokens = response_token_ids[0].squeeze()[input_ids.size(1):]
            gen_text = tokenizer.decode(gen_tokens,
                                              skip_special_tokens=True,
                                              clean_up_tokenization_spaces=True).strip()
            result = calculate_metrics(gen_text,
                                       sample['target'],
                                       sample['source'])
            results.append(result | {'gen_text': gen_text})
    return results


if __name__ == "__main__":
    set_seed(SEED)
    save_dir = "eval_results"
    os.makedirs(save_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description="Evaluating sft and policy model outputs.")
    parser.add_argument("--ckpt_path", type=str, help="path to sft or policy model checkpoint")
    args = parser.parse_args()

    if 'flant5' in args.ckpt_path:
        AutoModelForGeneration = AutoModelForCausalLM
        task_prefix = FLAN_T5_TASK_PREFIX
    else:
        AutoModelForGeneration = AutoModelForCausalLM
        task_prefix = TASK_PREFIX

    dataset = build_dataset(args.ckpt_path, task_prefix)

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)

    model = AutoModelForGeneration.from_pretrained(args.ckpt_path,
                                                   torch_dtype=torch.bfloat16)
    model.to(device)

    # evaluate with heuristic generation config
    heuristic_results = evaluate_model(model, dataset["test"],
                                       tokenizer, heuristic_generation_kwargs)

    heuristic_file_path = os.path.join(save_dir,
                                       args.ckpt_path.split("/")[-2] + "_heuristic.csv")
    with open(heuristic_file_path, mode="w", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=heuristic_results[0].keys())
        writer.writeheader()
        writer.writerows(heuristic_results)

    # evaluate with basic generation config
    basic_results = evaluate_model(model, dataset["test"],
                                   tokenizer, basic_generation_kwargs)
    basic_file_path = os.path.join(save_dir,
                                   args.ckpt_path.split("/")[-2] + "_basic.csv")

    with open(basic_file_path, mode="w", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=basic_results[0].keys())
        writer.writeheader()
        writer.writerows(basic_results)

    # print out results
    heuristic_avg_scores = {
        metric: np.mean([x[metric] for x in heuristic_results])
        for metric in heuristic_results[0].keys()
        if metric not in ["gen_text"]
    }
    print("Heuristic average scores:", heuristic_avg_scores)

    basic_avg_scores = {
        metric: np.mean([x[metric] for x in basic_results])
        for metric in basic_results[0].keys()
        if metric not in ["gen_text"]
    }
    print("Basic average scores:", basic_avg_scores)
