"""
This module calculates the statistics about the PNAS Scientific Abstract-Significance
Statement corpus.
"""

import heapq
import json
import pickle
from typing import Dict, List

import numpy as np
from datasets import load_from_disk
from nltk.tokenize import sent_tokenize
from sacremoses import MosesTokenizer
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from utils import (DATASET_PATH, VOA1500, WORD_ACCESSIBILITY_MODEL,
                   WORD_FREQ_CSV, compute_ari, compute_flesch_kincaid,
                   compute_sent_len, compute_token_accessibility,
                   read_token_frequencies)

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


def calculate_stats(text: str) -> Dict[str, float]:
    metrics_dict = {}
    text = text.strip()
    metrics_dict.update({"ari": compute_ari(text)})
    metrics_dict.update({"fk": compute_flesch_kincaid(text)})
    # complexity measure
    word_accessibility_list = []
    sent_len_list = []
    num_words = 0
    num_chars = 0
    num_voa_words = 0
    sents = sent_tokenize(text)
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


def calculate_mean_std(metrics_dicts: List[Dict[str, float]]) -> Dict[
    str, Dict[str, float]]:
    metrics = {
        "ari": [],
        "fk": [],
        "voa_log_ratio": [],
        "avg_sent_len": [],
        "avg_word_accessibility": [],
        "num_sents": [],
        "avg_word_len": []
    }
    for metrics_dict in metrics_dicts:
        for key in metrics.keys():
            metrics[key].append(metrics_dict[key])
    mean_std_dict = {}
    for key, values in metrics.items():
        mean_std_dict[key] = {
            "mean": np.mean(values),
            "std": np.std(values)
        }
    return mean_std_dict


if __name__ == "__main__":

    dataset = load_from_disk(DATASET_PATH)
    abstracts = []
    significances = []
    for split in ["train", "validation", "test"]:
        abstracts.extend(dataset[split]['source'])
        significances.extend(dataset[split]['target'])

    abstract_metrics_dicts = []
    for abstract in tqdm(abstracts):
        abstract_metrics_dicts.append(calculate_stats(abstract))

    significance_metrics_dicts = []
    for significance in tqdm(significances):
        significance_metrics_dicts.append(calculate_stats(significance))

    abstract_mean_std = calculate_mean_std(abstract_metrics_dicts)
    significance_mean_std = calculate_mean_std(significance_metrics_dicts)

    # print out the mean and standard deviation for each metric
    print("Abstract Metrics:")
    for key, stats in abstract_mean_std.items():
        print(f"{key}: Mean = {stats['mean']:.2f}, Std = {stats['std']:.2f}")
    # ari: Mean = 18.87, Std = 2.81
    # fk: Mean = 19.20, Std = 2.43
    # voa_log_ratio: Mean = -0.43, Std = 0.25
    # avg_sent_len: Mean = 25.42, Std = 4.85
    # avg_word_accessibility: Mean = 11.82, Std = 0.44
    # num_sents: Mean = 8.15, Std = 2.09
    # avg_word_len: Mean = 5.31, Std = 0.38

    print("\nSignificance statement Metrics:")
    for key, stats in significance_mean_std.items():
        print(f"{key}: Mean = {stats['mean']:.2f}, Std = {stats['std']:.2f}")
    # ari: Mean = 18.07, Std = 3.13
    # fk: Mean = 18.56, Std = 2.71
    # voa_log_ratio: Mean = -0.31, Std = 0.26
    # avg_sent_len: Mean = 23.87, Std = 5.26
    # avg_word_accessibility: Mean = 11.94, Std = 0.45
    # num_sents: Mean = 4.80, Std = 1.17
    # avg_word_len: Mean = 5.37, Std = 0.40

    # perform t-tests on each pair of metrics and adjust p-values
    p_values = []
    for key in abstract_mean_std.keys():
        abstract_values = [metrics_dict[key] for metrics_dict in abstract_metrics_dicts]
        significance_values = [metrics_dict[key] for metrics_dict in
                               significance_metrics_dicts]
        t_stat, p_value = ttest_ind(abstract_values, significance_values,
                                    equal_var=False)
        p_values.append(p_value)
        print(f"{key}: t-statistic = {t_stat:.2f}, p-value = {p_value}")
    # ari: t - statistic = 11.19, p - value = 8.39190160097395e-29
    # fk: t - statistic = 10.24, p - value = 1.9573758576642127e-24
    # voa_log_ratio: t - statistic = -21.00, p - value = 5.76630279160237e-95
    # avg_sent_len: t - statistic = 12.67, p - value = 2.3204681354299235e-36
    # avg_word_accessibility: t - statistic = -11.26, p - value = 3.666007169497688e-29
    # num_sents: t - statistic = 81.75, p - value = 0.0
    # avg_word_len: t - statistic = -7.07, p - value = 1.7227837392890036e-12

    # adjust the p-values using Bonferroni correction
    adjusted_p_values = multipletests(p_values, alpha=0.05, method='bonferroni')[1]
    print("\nAdjusted P-Values:")
    for key, adj_p_value in zip(abstract_mean_std.keys(), adjusted_p_values):
        print(f"{key}: adjusted p-value = {adj_p_value}")
    # ari: adjusted p-value = 5.874331120681766e-28
    # fk: adjusted p-value = 1.370163100364949e-23
    # voa_log_ratio: adjusted p-value = 4.036411954121659e-94
    # avg_sent_len: adjusted p-value = 1.6243276948009465e-35
    # avg_word_accessibility: adjusted p-value = 2.5662050186483814e-28
    # num_sents: adjusted p-value = 0.0
    # avg_word_len: adjusted p-value = 1.2059486175023025e-11
