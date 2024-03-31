import argparse
import csv
import os
from typing import Dict

import evaluate
import numpy as np
import torch
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

from utils import BASELINE_MODEL, SEED, TOP_P, build_dataset, compute_ari

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metric_bleu = BLEU()
metric_sari = evaluate.load("sari")
metric_rouge = evaluate.load("rouge")


def calculate_metrics(
    generated_text: str, target_text: str, source_text: str
) -> Dict[str, float]:
    """
    Compute common evaluation metrics for a given generated text against reference
    texts,  including 'bleu', 'sari', and 'rougeL'.

    Args:
        generated_text: The generated text to be evaluated.
        target_text: The reference text.
        source_text: The original source text.

    Returns:
        A dictionary where the keys are the metric names and the values are the
        corresponding computed metric scores.
    """

    metrics_dict = {}
    generated_texts = [generated_text.strip()]
    source_texts = [source_text.strip()]
    target_texts = [[target_text.strip()]]

    # ari
    metrics_dict.update({"ari": compute_ari(generated_texts[0])})
    # sacrebleu
    metrics_dict.update(
        {"bleu": metric_bleu.corpus_score(generated_texts, target_texts).score}
    )
    # sari
    metrics_dict.update(
        metric_sari.compute(
            sources=source_texts, predictions=generated_texts, references=target_texts
        )
    )
    # rougeL
    _rouge = metric_rouge.compute(predictions=generated_texts, references=target_texts)
    metrics_dict.update({"rougeL": _rouge["rougeL"]})

    return metrics_dict


if __name__ == "__main__":
    torch.manual_seed(SEED)

    parser = argparse.ArgumentParser(
        description="Evaluating baseline and policy model outputs."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="path to policy model checkpoint",
    )
    parser.add_argument("--output_file", type=str,
                        help="dir to save evaluation results")
    args = parser.parse_args()

    # get data
    dataset = build_dataset(
        model_name=BASELINE_MODEL,
        task_prefix="summarize, simplify, and contextualize: ",
    )
    # fixme
    tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL)
    if args.ckpt_path:
        model = T5ForConditionalGeneration.from_pretrained(args.ckpt_path)
    else:
        model = T5ForConditionalGeneration.from_pretrained(BASELINE_MODEL)

    model.to(device)
    model.eval()

    results = []
    with torch.no_grad():
        for example in tqdm(dataset["test"]):
            input_ids = example["input_ids"].to(device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            outputs = model.generate(
                input_ids, top_p=TOP_P, max_length=512, do_sample=True
            )
            output = tokenizer.decode(
                outputs[0], clean_up_tokenization_spaces=True, skip_special_tokens=True
            )
            print(output)
            result = calculate_metrics(
                generated_text=output,
                target_text=example["target"],
                source_text=example["source"],
            )
            result.update(
                {
                    "source": example["source"],
                    "target": example["target"],
                    "output": output.strip(),
                }
            )
            results.append(result)

    # dump results to the CSV file
    if args.output_file:
        save_dir = os.path.join("evaluation_results", args.output_file)
        os.makedirs(save_dir, exist_ok=True)
        with open(save_dir, mode="w", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # print average scores
    avg_scores = {
        metric: np.mean([x[metric] for x in results])
        for metric in results[0].keys()
        if metric not in ["source", "target", "output"]
    }
    print(avg_scores)
