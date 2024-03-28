import os
import argparse
import numpy as np
import torch
import wandb
from sacrebleu.metrics import BLEU
from sacremoses import MosesTokenizer
from datasets import load_dataset, load_from_disk
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer
from typing import List, Callable
from trl import AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer

SEED = 42
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def save_checkpoint(model,
                    step,
                    eval_score,
                    save_dir="ckpts"):
    """
    Save model checkpoint if ARI mean is lower than 16.

    Args:
        model: The policy model to be saved.
        step: Current step number in the training loop.
        eval_score: Eval scores of thehe current evaluation.
        save_dir: Base directory for saving checkpoints.
    """
    # check if ARI mean condition is met
    if np.mean(eval_score['ari']) <= 16 and np.mean(eval_score['sacrebleu']) >= 30:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir,
                                 f"model_step_{step}_ari_{eval_score['ari']:.2f}_scarebleu_{eval_score['sacrebleu']:.2f}")
        print(f"Checkpointing model at step {step} with \n"
              f"ARI mean {np.mean(eval_score['ari']):.2f}\n"
              f"Sacrebleu mean {np.mean(eval_score['sacrebleu']):.2f}.")
        model.save_pretrained(save_path)


def evaluate_model(model, dataset, tokenizer, compute_ari_func, num_samples=32):
    """
    This function evaluates the model's performance (ARI and SacreBLEU) on a subset of
    the given dataset.

    Args:
        model: The policy model to be evaluated.
        dataset: The validation dataset to use for evaluation.
        tokenizer: The tokenizer used for the model.
        compute_ari_func: A function that computes the ARI score from a string.
        num_samples: Number of samples to evaluate on. Defaults to 32.

    Returns:
        A dictionary containing the average ARI score, its standard deviation, and the
        average SacreBLEU score of the model on the dataset.
    """
    model.eval()
    device = next(model.parameters()).device
    model.to(device)
    bleu = BLEU()

    # use first num_samples from the dataset
    sampled_dataset = [dataset[i] for i in range(num_samples)]

    ari_scores = []
    bleu_scores = []
    with torch.no_grad():
        for batch in tqdm(sampled_dataset):
            query_tensors = batch["input_ids"].to(device)

            if query_tensors.dim() == 1:
                query_tensors = query_tensors.unsqueeze(0)

            response_tensors = model.generate(
                query_tensors,
                top_p=0.9,
                max_new_tokens=512,
                do_sample=True
            )
            responses = tokenizer.batch_decode(response_tensors.cpu(),
                                               clean_up_tokenization_spaces=True,
                                               skip_special_tokens=True)
            # Assuming 'targets' is a list of reference texts for BLEU calculation
            targets = [batch["target"]]  # You may need to adjust how you access targets

            for response, target in zip(responses, targets):
                ari_scores.append(compute_ari_func(response))
                bleu_scores.append(bleu.corpus_score([response], [[target]]).score)

    return {
        'ari': ari_scores,
        'sacrebleu': bleu_scores,
    }


def linear_schedule(optimizer, start_lr, end_lr, num_training_steps):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr
    set in the optimizer to end_lr.

    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        start_lr: The initial learning rate.
        end_lr: The final learning rate.
        num_training_steps: The number of steps over which to linearly decrease the
            learning rate.
    """

    def lr_lambda(current_step: int):
        # Compute the current scale factor
        scale = max(0, (num_training_steps - current_step) / num_training_steps)
        # Compute the scaled learning rate
        lr = end_lr + (start_lr - end_lr) * scale
        return lr

    return LambdaLR(optimizer, lr_lambda)


def build_dataset(
        dataset_name: str,
        model_name: str = "haining/sas_baseline",
        task_prefix: str = "summarize, simplify, and contextualize: ",
):
    """
    Build dataset for training with FLAN-T5. This function filters out too short samples
    and then extracts a specific number of samples for training.

    Args:
        model_name: SFT'ed model name.
        dataset_name: The name of the dataset to be loaded. SAS dataset will be loaded
            from local files and the arxiv is a specific version of it
            "gfissore/arxiv-abstracts-2021."
        task_prefix: The prefix to prepend to each abstract for task
        instruction.
        num_samples: The number of samples to be extracted from the
        dataset for training.

    Returns:
        DataLoader: The DataLoader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = load_from_disk("resources/scientific_abstract_simplification_corpus")
    ds = ds.rename_columns({"source": "query"})

    def tokenize(sample):
        # prepend the task-specific prefix
        input_text = task_prefix + sample["query"]
        input_ids = tokenizer.encode(
            input_text,
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        sample["input_ids"] = torch.tensor(input_ids)
        sample["query"] = tokenizer.decode(sample["input_ids"],
                                           skip_special_tokens=True,
                                           clean_up_tokenization_spaces=True)

        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def reward2tensor(responses: List[str],
                  compute_ari_func: Callable[[str], float],
                  normalize: bool = False) -> List[torch.Tensor]:
    """
    Process responses through the Automated Readability Index function to compute
    rewards, optionally normalize, clip, and convert them to tensors.

    Args:
        responses: A list of strings for which to compute the rewards.
        compute_ari_func: A function that computes the ARI score from a string.
        normalize: If True, normalize the rewards to have a mean of 0 and a standard
        deviation of 1. After normalization, rewards are clipped to be between -1 and 1
            for stability. Defaults to False.

    Returns:
        A list of tensors containing the processed rewards.
    """
    # calculate raw rewards using ARI function and invert them
    rewards = [compute_ari_func(r) * (-1.0) for r in responses]

    if normalize:
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        rewards = [(r - mean_reward) / (std_reward + 1e-9) for r in rewards]
        # clip normalized rewards to be between -1 and 1 for stability
        rewards = [max(min(r, 1.0), -1.0) for r in rewards]

    # convert the clipped rewards to tensors
    reward_tensors = [torch.tensor(r, dtype=torch.float32) for r in rewards]

    return reward_tensors


def compute_ari(text: str):
    """
    Compute the Automated Readability Index (ARI) for a given text.
    The ARI formula is: 4.71 * (characters/words) + 0.5 * (words/sentences) - 21.43
    Incomplete sentences (likely not ending in a period, exclamation, or question mark)
    are not considered.

    Args:
    text: A string of text to compute ARI.

    Returns:
        A list of tensors containing the processed rewards.
    """
    mt = MosesTokenizer(lang='en')
    sentences = sent_tokenize(text)
    words = mt.tokenize(text)

    # check if the last sentence is complete
    if sentences and not sentences[-1].endswith((".", "?", "!")):
        # Remove the last sentence if it is incomplete
        sentences = sentences[:-1]

    character_count = sum(len(word) for word in words)
    sentences_count = len(sentences)
    words_count = len(words)

    # avoid division by zero
    if sentences_count == 0 or words_count == 0:
        return 0

    # Apply the ARI formula
    ari_score = (
            4.71 * (character_count / words_count)
            + 0.5 * (words_count / sentences_count)
            - 21.43
    )

    # clip for stability (assuming a reasonable ARI range)
    ari_score = max(min(ari_score, 35.0), 2.0)

    return ari_score


if __name__ == "__main__":
    torch.manual_seed(SEED)

    parser = argparse.ArgumentParser(
        description="Rewriting complex scholarly abstracts to laymen.")
    parser.add_argument("--task_name", type=str,
                        default="Scholarly Abstract Simplification",
                        help="Experiment name for tracking")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="Initial learning rate for optimizer")
    parser.add_argument("--mini_batch_size", type=int, default=4,
                        help="Mini batch size for PPO updates")
    parser.add_argument("--ppo_epochs", type=int, default=1,
                        help="Number of optimization rollouts per batch of samples "
                             "during PPO training")
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--model_name", type=str,
                        default="haining/sas_baseline",
                        help="Model name on huggingface")
    parser.add_argument("--early_stopping", action="store_true",
                        help="Enable early stopping if KL divergence is too high")
    parser.add_argument("--target_kl", type=float, default=1.0,
                        help="Target KL divergence for early stopping")
    parser.add_argument("--use_score_scaling", action="store_true",
                        help="Enable score scaling")
    parser.add_argument("--use_score_norm", action="store_true",
                        help="Enable score normalization")
    parser.add_argument("--score_clip", type=float, default=None,
                        help="Value to clip the scores, use 'None' to disable")

    args = parser.parse_args()
    config_kwargs = vars(args)
    config = PPOConfig(
        log_with="wandb",
        **config_kwargs)

    # monitor with wandb
    wandb.init(project=config.task_name, config=config)
    # build dataset
    dataset = build_dataset('sas', config.model_name)
    # init SFT'ed models
    policy_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name)
    ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # optimizer and lr scheduler
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate)
    lr_scheduler = linear_schedule(optimizer,
                                   start_lr=args.learning_rate,
                                   end_lr=1e-6,
                                   num_training_steps=50)

    ppo_trainer = PPOTrainer(
        config=config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset["train"],
        data_collator=collator,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    generation_kwargs = {
        "min_length": 10,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": 400,
    }

    for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        response_tensors, ref_response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            generate_ref_response=True,
            **generation_kwargs
        )

        batch["response"] = tokenizer.batch_decode(response_tensors,
                                                   skip_special_tokens=True)
        batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors,
                                                       skip_special_tokens=True)

        rewards = reward2tensor(
            batch["response"], compute_ari
        )
        # ref rewards
        ref_rewards = reward2tensor(
            batch["ref_response"], compute_ari
        )
        batch["ref_rewards"] = ref_rewards
        batch["advantage"] = [p - r for p, r in zip(rewards, ref_rewards)]

        # execute a PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(
            stats,
            batch,
            rewards,
            columns_to_log=["query", "response", "ref_response", "ref_rewards",
                            "advantage"],
        )
        # evaluate on validation set after every N steps
        # fixme: add arg to argparse
        # if step % config.n_steps_per_eval == 0:
        if step % 5 == 0:
            eval_score = evaluate_model(
                model=policy_model,
                dataset=dataset["validation"],
                tokenizer=tokenizer,
                compute_ari_func=compute_ari,
                num_samples=32
            )
            wandb.log(eval_score)
            print(f"Step: {step}, Validation ARI: {np.mean(eval_score['ari']):.2f}")
            print(f"Step: {step}, Validation sacreBLEU: {np.mean(eval_score['sacrebleu']):.2f}")

            # save checkpoint if condition is met: ari<=16 and sacreBLEU >=30
            save_checkpoint(
                model=policy_model,
                step=step,
                eval_score=eval_score
            )
