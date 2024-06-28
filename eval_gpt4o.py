
import csv
import heapq
import json
import os
import pickle
from typing import Dict

import evaluate
import numpy as np
from datasets import load_from_disk
from nltk.tokenize import sent_tokenize
from openai import OpenAI
from sacrebleu.metrics import BLEU
from sacremoses import MosesTokenizer

from utils import (DATASET_PATH, RESPONSE_TEMP, SEED, TASK_PREFIX, VOA1500,
                   WORD_ACCESSIBILITY_MODEL, WORD_FREQ_CSV, compute_ari,
                   compute_flesch_kincaid, compute_sent_len,
                   compute_token_accessibility, read_token_frequencies)

# eval
device = 'cpu'
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
SAVE_DIR = "eval_results"
os.makedirs(SAVE_DIR, exist_ok=True)


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


if __name__ == "__main__":
    # get gpt4o outputs
    print('*' * 90)
    print('Getting and Evaluating GP4O generation')
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "Set up OPENAI API Key"))
    MODEL = 'gpt-4o-2024-05-13'
    # system prompt adopted from
    # https://platform.openai.com/docs/guides/text-generation/json-mode
    SYSTEM_PROMPT = 'You are a helpful assistant designed to output JSON with a single key "abstract".'

    ds = load_from_disk(DATASET_PATH)
    generated_texts = []
    for abstract in ds['test']['source'][:5]:
        query = TASK_PREFIX + abstract + RESPONSE_TEMP
        response = client.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.01 + 1e-7,  # fair comparison
            max_tokens=256,
            seed=SEED,
            n=1
        )
        generated_text = json.loads(response.choices[0].message.content)['abstract']
        generated_texts.append(generated_text)

    # evaluate the generated texts using the function `calculate_metrics`
    results = []
    for generated_text, target_text, source_text in zip(generated_texts,
                                                        ds['test']['target'][:5],
                                                        ds['test']['source'][:5]):
        metrics = calculate_metrics(generated_text, target_text, source_text)
        results.append(metrics | {'generated_text': generated_text})

    # save the generated texts and metrics to a CSV file
    file_path = os.path.join(SAVE_DIR, f'{MODEL}.csv')
    with open(file_path, mode="w", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # print out the metrics
    avg_scores = {f"avg_{metric}": np.mean([x[metric] for x in results]) for metric in results[0].keys() if metric not in ["generated_text"]}
    std_scores = {f"std_{metric}": np.std([x[metric] for x in results]) for metric in results[0].keys() if metric not in ["generated_text"]}

    print(f'Metrics for {MODEL}:')
    print("Average scores:", avg_scores)
    print("Standard deviation of scores:", std_scores)
    print('*' * 90)
