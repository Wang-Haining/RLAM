import argparse
import csv
import os
from typing import Dict

import evaluate
import numpy as np
import torch
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import SEED, TOP_P, build_dataset, compute_ari, MODEL_NAME, RESPONSE_TEMP

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
        description="Evaluating sft and policy model outputs."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="path to sft or policy model checkpoint",
    )
    parser.add_argument("--output_file", type=str,
                        help="dir to save evaluation results")
    args = parser.parse_args()

    dataset = build_dataset()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(args.ckpt_path,
                                                 torch_dtype=torch.bfloat16)

    model.to(device)
    model.eval()

    results = []
    with torch.no_grad():
        for example in tqdm(dataset["test"]):
            input_ids = example["input_ids"].to(device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            # calculate the length of the input part
            input_length = input_ids.size(1)

            outputs = model.generate(
                input_ids,
                top_p=TOP_P,
                max_length=1024,
                do_sample=True,
                return_dict_in_generate=True,
                num_return_sequences=5,
            )

            # Decode only the newly generated tokens
            # Adjust the slicing as needed, here we skip the input tokens
            gen_tokens = outputs.sequences[:, input_length:].squeeze()
            output = tokenizer.decode(
                gen_tokens,
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True
            )
            # outputs = model.generate(
            #     input_ids, top_p=TOP_P, max_length=1024, do_sample=True
            # )
            # output = tokenizer.decode(
            #     outputs[0], clean_up_tokenization_spaces=True, skip_special_tokens=True
            # )
            result = calculate_metrics(
                generated_text=output,
                target_text=example["target"],
                source_text=example["query"],
            )
            result.update(
                {
                    "source": example["query"],
                    "target": example["target"],
                    "output": output.strip(),
                }
            )
            results.append(result)

    if args.output_file:
        save_dir = "evaluation_results"
        os.makedirs(save_dir, exist_ok=True)
        full_file_path = os.path.join(save_dir, args.output_file)
        # Write the results to the CSV file
        with open(full_file_path, mode="w", encoding="utf-8") as file:
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
