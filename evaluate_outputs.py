import argparse
import csv
import os
from typing import Dict, List

import evaluate
import numpy as np
import torch
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          AutoTokenizer)
from trl import set_seed

from utils import (FLAN_T5_TASK_PREFIX, FLANT5, GEMMA, OLMO, SEED, TASK_PREFIX,
                   build_dataset, compute_ari)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metric_bleu = BLEU()
metric_sari = evaluate.load("sari")
metric_rouge = evaluate.load("rouge")
metric_bertscore = evaluate.load("bertscore")


def calculate_metrics(generated_text: str, target_text: str, source_text: str) -> Dict[str, float]:
    metrics_dict = {}
    generated_texts = [generated_text.strip()]
    source_texts = [source_text.strip()]
    target_texts = [[target_text.strip()]]

    metrics_dict.update({"ari": compute_ari(generated_texts[0])})
    metrics_dict.update({"bleu": metric_bleu.corpus_score(generated_texts, target_texts).score})
    metrics_dict.update(metric_sari.compute(sources=source_texts, predictions=generated_texts, references=target_texts))
    _rouge = metric_rouge.compute(predictions=generated_texts, references=target_texts)
    metrics_dict.update({"rougeL": _rouge["rougeL"]})
    bertscore_result = metric_bertscore.compute(predictions=generated_texts, references=target_texts, lang="en", device="cpu")
    metrics_dict.update({"bertscore": np.mean(bertscore_result["f1"])})

    return metrics_dict


def evaluate_model(model, dataset: List[Dict], tokenizer, generation_kwargs) -> List[Dict]:
    results = []
    model.eval()
    with torch.no_grad():
        for start_idx in tqdm(range(0, len(dataset), 4)):
            batch = dataset[start_idx:start_idx + 4]
            print(f"Batch structure: {batch}")  # Debug print

            try:
                input_ids = torch.stack([example["input_ids"] for example in batch]).to(device)
            except Exception as e:
                print(f"Error in stacking input_ids: {e}")
                print(f"Batch content: {batch}")
                continue

            input_length = input_ids.size(1)
            outputs = model.generate(input_ids, **generation_kwargs)
            gen_tokens = outputs[:, input_length:]
            outputs = [tokenizer.decode(tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True) for tokens in gen_tokens]

            for example, output in zip(batch, outputs):
                result = calculate_metrics(generated_text=output, target_text=example["target"], source_text=example["query"])
                result.update({"source": example["query"], "target": example["target"], "output": output.strip()})
                results.append(result)
                print(f"Result: {result}")  # Debug print
    return results


if __name__ == "__main__":
    set_seed(SEED)

    parser = argparse.ArgumentParser(description="Evaluating sft and policy model outputs.")
    parser.add_argument("--ckpt_path", type=str, help="path to sft or policy model checkpoint")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Exponential penalty to the length")
    args = parser.parse_args()

    if 'gemma' in args.ckpt_path:
        AutoModelForGeneration = AutoModelForCausalLM
        model_name = GEMMA
        task_prefix = TASK_PREFIX
    elif 'olmo' in args.ckpt_path.lower():
        AutoModelForGeneration = AutoModelForCausalLM
        model_name = OLMO
        task_prefix = TASK_PREFIX
    elif 't5' in args.ckpt_path:
        AutoModelForGeneration = AutoModelForSeq2SeqLM
        model_name = FLANT5
        task_prefix = FLAN_T5_TASK_PREFIX
    else:
        raise ValueError(f"Unknown sft'ed ckpt path {args.ckpt_path}")

    dataset = build_dataset(model_name, task_prefix)
    print(f"Dataset length: {len(dataset['test'])}")  # Debug print to check dataset length

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForGeneration.from_pretrained(args.ckpt_path, torch_dtype=torch.bfloat16)
    model.to(device)

    # heuristic generation config
    heuristic_generation_kwargs = {
        "top_p": 0.9,
        "max_length": 768,
        "num_beams": 4,
        "length_penalty": args.length_penalty,
        "do_sample": True,
        "return_dict_in_generate": True,
        "num_return_sequences": 1,
    }

    # basic generation config
    basic_generation_kwargs = {
        "max_length": 768,
        "do_sample": True,
        "return_dict_in_generate": True,
        "num_return_sequences": 1,
    }

    # evaluate with heuristic generation config
    heuristic_results = evaluate_model(model, dataset["test"], tokenizer, heuristic_generation_kwargs)
    print(f"Heuristic results length: {len(heuristic_results)}")  # Debug print

    save_dir = "eval_results"
    os.makedirs(save_dir, exist_ok=True)

    if heuristic_results:
        heuristic_file_path = os.path.join(save_dir, args.ckpt_path.split("/")[-2] + "_heuristic.csv")
        with open(heuristic_file_path, mode="w", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=heuristic_results[0].keys())
            writer.writeheader()
            writer.writerows(heuristic_results)
    else:
        print("No heuristic results to save.")

    # evaluate with basic generation config
    basic_results = evaluate_model(model, dataset["test"], tokenizer, basic_generation_kwargs)
    print(f"Basic results length: {len(basic_results)}")  # Debug print

    if basic_results:
        basic_file_path = os.path.join(save_dir, args.ckpt_path.split("/")[-2] + "_basic.csv")
        with open(basic_file_path, mode="w", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=basic_results[0].keys())
            writer.writeheader()
            writer.writerows(basic_results)
    else:
        print("No basic results to save.")

    if heuristic_results:
        heuristic_avg_scores = {
            metric: np.mean([x[metric] for x in heuristic_results])
            for metric in heuristic_results[0].keys()
            if metric not in ["source", "target", "output"]
        }
        print("Heuristic average scores:", heuristic_avg_scores)

    if basic_results:
        basic_avg_scores = {
            metric: np.mean([x[metric] for x in basic_results])
            for metric in basic_results[0].keys()
            if metric not in ["source", "target", "output"]
        }
        print("Basic average scores:", basic_avg_scores)
