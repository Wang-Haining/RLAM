"""
This module implements evaluation functions for abstracts (to be simplified).
"""

import heapq
import json
import pickle
from typing import Dict

import numpy as np
from datasets import load_from_disk
from nltk.tokenize import sent_tokenize
from sacremoses import MosesTokenizer
from tqdm import tqdm

from utils import (DATASET_PATH, VOA1500, WORD_ACCESSIBILITY_MODEL,
                   WORD_FREQ_CSV, compute_ari, compute_flesch_kincaid,
                   compute_sent_len, compute_token_accessibility,
                   read_token_frequencies)

# get word frequencies and the model to predict relative rare word's accessibility
token_freq = read_token_frequencies(WORD_FREQ_CSV)
top_100k_tokens = heapq.nlargest(100000, token_freq, key=token_freq.get)
# load for making predictions word accessibility
wa_model = pickle.load(open(WORD_ACCESSIBILITY_MODEL, "rb"))
total_tokens = sum(token_freq.values())
mt = MosesTokenizer(lang="en")
# VOA Word Book, Section A-Z, Science programs, and Organs of the body (1517 in total)
# from https://simple.wikipedia.org/wiki/Wikipedia:VOA_Special_English_Word_Book
# scraped on May 15, 2024
voa1500 = json.load(open(VOA1500, "r", encoding="utf-8"))


def calculate_metrics_for_abstract(source_text: str) -> Dict[str, float]:
    metrics_dict = {}
    metrics_dict.update({"ari": compute_ari(source_text)})
    metrics_dict.update({"fk": compute_flesch_kincaid(source_text)})
    # complexity measure
    word_accessibility_list = []
    sent_len_list = []
    num_words = 0
    num_chars = 0
    num_voa_words = 0
    sents = sent_tokenize(source_text)
    for sent in sents:
        sent_len_list.append(compute_sent_len(sent))
        for token in mt.tokenize(sent, escape=False):
            num_words += 1
            num_chars += len(token)
            if token.lower() in voa1500:
                num_voa_words += 1
            word_accessibility_list.append(
                compute_token_accessibility(
                    token, top_100k_tokens, wa_model, total_tokens, token_freq
                )
            )
    p = (num_voa_words / num_words) + 1e-12
    metrics_dict.update({"voa_log_ratio": np.log(p / (1 - p))})
    metrics_dict.update({"avg_sent_len": np.mean(sent_len_list)})
    metrics_dict.update({"avg_word_accessibility": np.mean(word_accessibility_list)})
    metrics_dict.update({"num_sents": len(sents)})
    metrics_dict.update({"avg_word_len": num_chars / num_words})
    return metrics_dict


if __name__ == "__main__":
    ds = load_from_disk(DATASET_PATH)['test']

    eval_results = []
    for t in tqdm(ds['source']):
        eval_results.append(calculate_metrics_for_abstract(t))
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
    # print out results
    print("*" * 90)
    print("Metrics on original abstracts:")
    print("Average scores: {}".format(avg_scores))
    print("Standard deviation of scores: {}".format(std_scores))
    print("*" * 90)
