"""
This module implements the Reinforcement Learning from Uncombined Accessibility Measures
(RLUAM).
"""

import argparse
import heapq
import os
import pickle
import shutil
from typing import List

import numpy as np
import torch
import wandb
from nltk.tokenize import sent_tokenize
from sacrebleu.metrics import BLEU
from sacremoses import MosesTokenizer
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed
from trl import (AutoModelForCausalLMWithValueHead,
                 AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer)

from utils import (CLM_MODEL_NAME, PROJECT_NAME, SEED,
                   WORD_ACCESSIBILITY_MODEL, WORD_FREQ_CSV, build_dataset,
                   collator, compute_ari, compute_sent_len,
                   compute_token_accessibility, read_token_frequencies)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

# get word frequencies and the model to predict relative rare word's accessibility
token_freq = read_token_frequencies(WORD_FREQ_CSV)
top_100k_tokens = heapq.nlargest(100000, token_freq, key=token_freq.get)
# load for making predictions word accessibility
wa_model = pickle.load(open(WORD_ACCESSIBILITY_MODEL, 'rb'))
total_tokens = sum(token_freq.values())


def save_checkpoint(model, epoch, step, eval_score, num_saved_ckpts, save_folder):
    """
    Save model checkpoint if it's among the ones with the lowest ARI scores.

    Args:
        model: The policy model to be saved.
        epoch: Current epoch number in the training loop.
        step: Current step number in the training loop.
        eval_score: Eval scores of the current evaluation.
        num_saved_ckpts: Number of saved best checkpoints.
        save_folder: Directory for saving checkpoints, under directory `ckpts`.
    """
    save_dir = os.path.join("ckpts", save_folder)
    os.makedirs(save_dir, exist_ok=True)
    metadata_path = os.path.join(save_dir, "metadata.npz")

    if os.path.exists(metadata_path):
        metadata = np.load(metadata_path, allow_pickle=True)
        saved_models = list(metadata["saved_models"])
        print("Loaded saved models:", saved_models)
    else:
        saved_models = []

    current_ari_mean = np.mean(eval_score["ari"])
    save_path = os.path.join(save_dir,
                    f"model_epoch_{epoch}_step_{step}_ari_{current_ari_mean:.2f}.pt")
    print("Current ARI Mean:", current_ari_mean)

    # save the first three and remove the worst when a good one comes alone
    if (len(saved_models) < num_saved_ckpts) or (
            current_ari_mean < max(m['ari_mean'] for m in saved_models)):
        model.save_pretrained(save_path)
        saved_models.append({
            "path": save_path,
            "ari_mean": current_ari_mean,
            "epoch": epoch,
            "step": step
        })

        saved_models.sort(key=lambda x: x["ari_mean"])
        worst_model = saved_models.pop()
        shutil.rmtree(worst_model["path"])

    np.savez(metadata_path, saved_models=saved_models)


def evaluate_model(model, dataset, tokenizer, num_samples=32):
    """
    This function evaluates the model's performance (ARI and SacreBLEU) on a subset of
    the given dataset.

    Args:
        model: The policy model to be evaluated.
        dataset: The validation dataset to use for evaluation.
        tokenizer: The tokenizer used for the model.
        num_samples: Number of samples to evaluate on. Defaults to 32.

    Returns:
        A dictionary containing ARI scores, its standard deviation, and the
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
    avg_sent_len = []
    avg_word_access = []
    with torch.no_grad():
        for batch in tqdm(sampled_dataset):
            query_tensors = batch["input_ids"].to(device)

            if query_tensors.dim() == 1:
                query_tensors = query_tensors.unsqueeze(0)
            response_tensors = model.generate(query_tensors,
                                              top_k=0.0,  # fixme: TOP_K
                                              top_p=1.0,  # fixme: TOP_P
                                              max_new_tokens=512,
                                              do_sample=True)
            responses = tokenizer.batch_decode(
                response_tensors.cpu(),
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True,
            )
            targets = [batch["target"]]

            for response, target in zip(responses, targets):
                ari_scores.append(compute_ari(response))
                bleu_scores.append(bleu.corpus_score([response], [[target]]).score)
                rewards = compute_rewards([response])
                avg_sent_len.append(rewards['sl_reward'][0])
                avg_word_access.append(rewards['wa_reward'][0])

    return {
        "ari": ari_scores,
        "sacrebleu": bleu_scores,
        "avg_sent_len": avg_sent_len,
        "avg_word_accessibility": avg_word_access
    }


def compute_rewards(responses: List[str],
                    compute_sent_len=compute_sent_len,
                    compute_token_accessibility=compute_token_accessibility,
                    top_100k_tokens=top_100k_tokens,
                    wa_model=wa_model,
                    total_tokens=total_tokens,
                    token_freq=token_freq) -> (List[torch.Tensor], List[torch.Tensor]):
    """
    Calculate rewards for a batch of responses:
    - Avg sentence length: Computed over all sentences in a response. (Note, because we
        want average to be shorter, we need to negate sentence length in order to
        maximize the rewards.)
    - Avg word accessibility: Defined as the negative logarithm of the token frequency
      per billion, based on its occurrences in the English Wikipedia corpus.

    During pilot running, models cheat by spitting out only one eos token. So we
    penalize both word accessibility and sentence length with reasonably large negative
    feedback in such situations.

    Parameters:
        responses: A list of response strings to process.
        compute_sent_len: A function to compute the length of a sentence.
        compute_token_accessibility: A function to compute the accessibility of a token.
        top_100k_tokens: Set of the top 100k tokens based on frequency.
        wa_model: The model used to estimate word accessibility.
        total_tokens: Total number of tokens in the reference corpus.
        token_freq: Dictionary of token frequencies.

    Returns:
        A tuple containing two lists of tensors:
        - Sentence length rewards.
        - Word accessibility rewards.
    """
    sent_len_rewards = []
    word_accessibility_rewards = []
    mt = MosesTokenizer(lang='en')
    for response in responses:
        # EOS tokens of gemma and t5
        if (response.strip() in ['<eos>', '</s>']) or (len(response.strip()) <= 10):
            sent_len_rewards.append(50.0)
            word_accessibility_rewards.append(2.0)
        else:
            sent_len_list = []
            word_accessibility_list = []
            sents = sent_tokenize(response)
            for sent in sents:
                sent_len_list.append(compute_sent_len(sent))
                for token in mt.tokenize(sent):
                    word_accessibility_list.append(
                        compute_token_accessibility(token,
                                                 top_100k_tokens,
                                                 wa_model,
                                                 total_tokens,
                                                 token_freq))
            sent_len_rewards.append(np.mean(sent_len_list))
            word_accessibility_rewards.append(np.mean(word_accessibility_list))
    # negate sentence length for intuitive reward maximization
    sent_len_rewards = [-1.0 * torch.tensor(r, dtype=torch.float32) for
                        r in sent_len_rewards]
    word_accessibility_rewards = [torch.tensor(r, dtype=torch.float32) for
                               r in word_accessibility_rewards]
    return {"sl_reward": sent_len_rewards,
            "wa_reward": word_accessibility_rewards}


if __name__ == "__main__":
    set_seed(SEED)

    # fmt: off
    parser = argparse.ArgumentParser(description="Rewriting complex scholarly abstracts to laymen.")
    # ppo_config relevant
    parser.add_argument("--seed", type=int, default=SEED, help="Batch size for training")
    parser.add_argument("--steps", type=int, default=2000000, help="Number of training steps. A upper bound of total training steps. See num_epochs for details.")
    parser.add_argument("--learning_rate", type=float, default=3e-6, help="Adam learning rate")
    # kl objective
    # todo: adap_kl_ctrl
    parser.add_argument("--adap_kl_ctrl", type=lambda x: (str(x).lower() == 'true'), default=True, help="Use adaptive KL control, otherwise linear")
    parser.add_argument("--init_kl_coef", type=float, default=0.2, help="Initial KL penalty coefficient (used for adaptive and linear control). See formula (2) in https://arxiv.org/pdf/1909.08593")
    parser.add_argument("--kl_penalty", type=str, choices=['kl', 'abs', 'mse', 'full'], default="kl", help="KL penalty options")
    parser.add_argument("--target", type=float, default=6, help="Target KL value for adaptive KL control")
    parser.add_argument("--horizon", type=float, default=10000, help="Horizon for adaptive KL control")
    # ppo
    parser.add_argument("--lam", type=float, default=0.95, help="Lambda parameter for advantage calculation")
    parser.add_argument("--cliprange", type=float, default=0.2, help="Range for clipping in PPO policy gradient loss")
    parser.add_argument("--cliprange_value", type=float, default=0.2, help="Range for clipping values in loss calculation")
    parser.add_argument("--vf_coef", type=float, default=0.1, help="Scaling factor for value loss")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--mini_batch_size", type=int, default=2, help="Mini batch size for PPO updates")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--ppo_epochs", type=int, default=2, help="Number of optimisation epochs per batch of samples")
    parser.add_argument("--max_grad_norm", type=float, default=None, help="Maximum gradient norm for gradient clipping")
    parser.add_argument("--early_stopping", type=lambda x: (str(x).lower() == 'true'), default=False, help="Whether to stop the PPO optimization loop early if KL is too high")
    parser.add_argument("--target_kl", type=float, default=1.0, help="Stop early if we exceed this value by over 50%")
    parser.add_argument("--compare_steps", type=int, default=1, help="Number of steps between comparison of the current reward with the best seen so far")
    parser.add_argument("--ratio_threshold", type=float, default=10.0, help="Skip mini-batches with high PPO ratios that can cause loss spikes")
    # for rewards, following https://arxiv.org/pdf/1909.08593
    parser.add_argument("--use_score_scaling", action="store_true", help="Enable score scaling")
    parser.add_argument("--use_score_norm", action="store_true", help="Enable score normalization")
    parser.add_argument("--score_clip", type=float, default=None, help="Value to clip the scores, use 'None' to disable")
    parser.add_argument("--whiten_rewards", action='store_true', help="Whiten the rewards before computing advantages")
    # misc
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of total epochs")
    parser.add_argument("--sl_coef", type=float, default=1.0, help="Scaling factor for sentence length reward (will keep this frozen as 1.0)")
    parser.add_argument("--wa_coef", type=float, default=1.0, help="Scaling factor for word accessibility reward (will vary it for an optimal value)")
    parser.add_argument("--max_new_tokens", type=int, default=300, help="Max new tokens in rollouts.")
    parser.add_argument("--skip_special_tokens", type=lambda x: (str(x).lower() == 'true'), default=False, help="Whether to skip special tokens when decoding rollouts")
    parser.add_argument("--eval_interval", type=int, default=10, help="Interval between evaluations")
    parser.add_argument("--num_eval_samples", type=int, default=32, help="Num of samples for evaluation")
    parser.add_argument("--num_saved_ckpts", type=int, default=3, help="Num of best ckpts to save")
    parser.add_argument("--save_folder", type=str, default=f"ppo_{CLM_MODEL_NAME}", help="Experiment name for checkpointing, under the directory of ckpts")
    parser.add_argument("--sft_ckpt_path", type=str, help="Path to the SFT'ed model")

    args = parser.parse_args()
    # ignore the extra args not for ppo
    config_kwargs = vars(args).copy()
    keys_to_pop = ["num_epochs", "sl_coef", "wa_coef", "max_new_tokens",
                   "skip_special_tokens", "eval_interval", "num_eval_samples",
                   "num_saved_ckpts", "save_folder", "sft_ckpt_path"]
    for key in keys_to_pop:
        config_kwargs.pop(key, None)
    # fmt: on
    config = PPOConfig(log_with="wandb", **config_kwargs)
    # monitor with wandb
    run_name = "ppo_" + args.sft_ckpt_path.split("/")[-2].split('_')[-1]
    wandb.init(project=PROJECT_NAME, name=run_name, config=args)

    # build dataset
    dataset = build_dataset(model_name=args.sft_ckpt_path)

    # init SFT'ed models
    if 'gemma' in args.sft_ckpt_path:
        AutoModelForLMWithValueHead = AutoModelForCausalLMWithValueHead
    elif 't5' in args.sft_ckpt_path:
        AutoModelForLMWithValueHead = AutoModelForSeq2SeqLMWithValueHead
    else:
        raise ValueError(f"Unknown sft'ed ckpt path {args.sft_ckpt_path}")
    policy_model = AutoModelForLMWithValueHead.from_pretrained(
        args.sft_ckpt_path, torch_dtype=torch.bfloat16
    )
    ref_model = AutoModelForLMWithValueHead.from_pretrained(
        args.sft_ckpt_path, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.sft_ckpt_path)

    # init optimizer
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate)

    ppo_trainer = PPOTrainer(
        config=config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset["train"],
        data_collator=collator,
        optimizer=optimizer,
    )

    rollout_kwargs = {
        "min_length": -1 if 'gemma' in args.sft_ckpt_path else 2,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": args.max_new_tokens,
    }

    for epoch in range(args.num_epochs):
        for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            query_tensors = batch["input_ids"]
            response_tensors, ref_response_tensors = ppo_trainer.generate(
                query_tensors,
                return_prompt=False,
                generate_ref_response=True,
                **rollout_kwargs,
            )

            batch["response"] = tokenizer.batch_decode(response_tensors,
                                        skip_special_tokens=args.skip_special_tokens)
            batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors,
                                        skip_special_tokens=args.skip_special_tokens)
            # calculate and balance rewards
            rewards = compute_rewards(batch["response"])
            rewards = [args.sl_coef * sl + args.wa_coef * wa for
                       sl, wa in zip(rewards['sl_reward'], rewards['wa_reward'])]

            # ref rewards
            ref_rewards = compute_rewards(batch["ref_response"])
            ref_rewards = [args.sl_coef * sl + args.wa_coef * wa for
                           sl, wa in zip(ref_rewards['sl_reward'],
                                         ref_rewards['wa_reward'])]
            batch["ref_rewards"] = ref_rewards
            batch["advantage"] = [p - r for p, r in zip(rewards, ref_rewards)]

            # execute a PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(
                stats,
                batch,
                rewards,
                columns_to_log=["query", "response", "ref_response",
                                "ref_rewards", "advantage"],
            )
            wandb.log(rollout_kwargs)

            # evaluate on validation set after every n steps
            if step % args.eval_interval == 0:
                eval_score = evaluate_model(
                    model=policy_model,
                    dataset=dataset["validation"],
                    tokenizer=tokenizer,
                    num_samples=args.num_eval_samples,
                )
                wandb.log(eval_score)
                _sent_len_reward = np.mean(eval_score['avg_sent_len'])
                _word_accessibility_reward = np.mean(eval_score['avg_word_accessibility'])
                _total_reward = args.sl_coef * _sent_len_reward + args.wa_coef * _word_accessibility_reward
                print(f"Epoch-Step: {epoch}-{step}, Eval ARI: {np.mean(eval_score['ari']):.2f}")
                print(f"Epoch-Step: {epoch}-{step}, Eval BLEU: {np.mean(eval_score['sacrebleu']):.2f}")
                print(f"Epoch-Step: {epoch}-{step}, Eval avg. neg. sentence Length: {_sent_len_reward:.2f}")
                print(f"Epoch-Step: {epoch}-{step}, Eval avg. word accessibility: {_word_accessibility_reward:.2f}")
                print(f"Epoch-Step: {epoch}-{step}, Eval total rewards: {_total_reward:.2f}")
                wandb.log({
                    "Eval/Epoch": epoch,
                    "Eval/Step": step,
                    "Eval/ARI": np.mean(eval_score['ari']),
                    "Eval/BLEU": np.mean(eval_score['sacrebleu']),
                    "Eval/avg. neg. sentence Length": _sent_len_reward,
                    "Eval/avg. word accessibility": _word_accessibility_reward,
                    "Eval/total rewards": _total_reward
                })
                # save top-3 checkpoints wrt ARI and their metadata
                save_checkpoint(
                    model=policy_model,
                    epoch=epoch,
                    step=step,
                    eval_score=eval_score,
                    num_saved_ckpts=args.num_saved_ckpts,
                    save_folder=args.save_folder)
