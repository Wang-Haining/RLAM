import argparse
import heapq
import os
import pickle
from typing import Callable, List

import numpy as np
import torch
import wandb
from nltk.tokenize import sent_tokenize
from sacrebleu.metrics import BLEU
from sacremoses import MosesTokenizer
from tqdm import tqdm
from transformers import AutoTokenizer
from trl import (AutoModelForCausalLMWithValueHead,
                 AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer)

from utils import (CLM_MODEL_NAME, PROJECT_NAME, SEED, TOP_P,
                   WORD_DIFFICULTY_MODEL, WORD_FREQ_CSV, ByteNGramExtractor,
                   build_dataset, collator, compute_sent_len,
                   compute_token_difficulty, create_dataframe, custom_analyzer,
                   define_transformers, read_token_frequencies, reshape_data)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# get word frequencies and the model to predict relative rare word's difficulty
token_freq = read_token_frequencies(WORD_FREQ_CSV)
top_100k_tokens = heapq.nlargest(100000, token_freq, key=token_freq.get)
wd_model = pickle.load(open(WORD_DIFFICULTY_MODEL, 'rb'))
total_tokens = sum(token_freq.values())

# compute_token_difficulty('good', top_100k_tokens, wd_model, total_tokens, token_freq)


def save_checkpoint(model, step, eval_score, save_folder):
    """
    Save model checkpoint if it's among the three with the lowest ARI scores and always
    save the last model.

    Args:
        model: The policy model to be saved.
        step: Current step number in the training loop.
        eval_score: Eval scores of the current evaluation.
        save_folder: Directory for saving checkpoints, under directory `ckpts`.
    """
    save_dir = os.path.join("ckpts", save_folder)
    os.makedirs(save_dir, exist_ok=True)

    # define the path for storing metadata about saved models and the last model
    metadata_path = os.path.join(save_dir, "metadata.npz")
    last_model_path = os.path.join(save_dir, f"last_model_step_{step}.pt")

    # load or initialize metadata
    if os.path.exists(metadata_path):
        metadata = np.load(metadata_path, allow_pickle=True)
        saved_models = list(metadata["saved_models"])
    else:
        saved_models = []

    current_ari_mean = np.mean(eval_score["ari"])
    model.save_pretrained(last_model_path)  # Save the most recent model
    print(f"Saved the most recent model at step {step}.")

    save_path = os.path.join(
        save_dir,
        f"model_step_{step}_ari_{current_ari_mean:.2f}.pt",
    )

    if (len(saved_models) < 3 or current_ari_mean < max(saved_models,
                                                        key=lambda x: x["ari_mean"])["ari_mean"]):
        print(f"Saving model at step {step} with ARI mean {current_ari_mean:.2f}.")
        model.save_pretrained(save_path)
        saved_models.append({"path": save_path, "ari_mean": current_ari_mean})

        # remove the worst model if more than three are saved
        if len(saved_models) > 3:
            worst_model = max(saved_models, key=lambda x: x["ari_mean"])
            saved_models.remove(worst_model)
            if os.path.isfile(worst_model["path"]):
                os.remove(worst_model["path"])
                print(f"Removed model with ARI mean {worst_model['ari_mean']:.2f}.")
            else:
                print(f"Error: Attempted to remove a directory or non-existent file: "
                      f"{worst_model['path']}")

    else:
        print(f"Model at step {step} with ARI mean {current_ari_mean:.2f} "
              f"not saved as a top model.")

    np.savez(metadata_path, saved_models=saved_models)


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
                query_tensors, top_p=TOP_P, max_new_tokens=512, do_sample=True
            )
            responses = tokenizer.batch_decode(
                response_tensors.cpu(),
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True,
            )
            targets = [batch["target"]]

            for response, target in zip(responses, targets):
                ari_scores.append(compute_ari_func(response))
                bleu_scores.append(bleu.corpus_score([response], [[target]]).score)

    return {
        "ari": ari_scores,
        "sacrebleu": bleu_scores,
    }


def compute_rewards(responses: List[str],
                    compute_sent_len=compute_sent_len,
                    compute_token_difficulty=compute_token_difficulty,
                    top_100k_tokens=top_100k_tokens,
                    wd_model=wd_model,
                    total_tokens=total_tokens,
                    token_freq=token_freq) -> (List[torch.Tensor], List[torch.Tensor]):
    """
    Calculate rewards for a batch of responses:
    - Avg sentence length: Computed over all sentences in a response.
    - Avg word difficulty: Defined as the negative logarithm of the token frequency
      per billion, based on its occurrences in the English Wikipedia corpus.

    Parameters:
        responses: A list of response strings to process.
        compute_sent_len: A function to compute the length of a sentence.
        compute_token_difficulty: A function to compute the difficulty of a token.
        top_100k_tokens: Set of the top 100k tokens based on frequency.
        wd_model: The model used to estimate word difficulty.
        total_tokens: Total number of tokens in the reference corpus.
        token_freq: Dictionary of token frequencies.

    Returns:
        A tuple containing two lists of tensors:
        - Sentence length rewards.
        - Word difficulty rewards.
    """
    sent_len_rewards = []
    word_difficulty_rewards = []
    mt = MosesTokenizer(lang='en')
    for response in responses:
        sent_len_list = []
        word_difficulty_list = []
        sents = sent_tokenize(response)
        for sent in sents:
            sent_len_list.append(compute_sent_len(sent))
            for token in mt.tokenize(sent):
                word_difficulty_list.append(compute_token_difficulty(token,
                                                                     top_100k_tokens,
                                                                     wd_model,
                                                                     total_tokens,
                                                                     token_freq))
        sent_len_rewards.append(np.mean(sent_len_list))
        word_difficulty_rewards.append(np.mean(word_difficulty_list))
    sent_len_reward_tensors = [torch.tensor(r, dtype=torch.float32) for r in
                               sent_len_rewards]
    word_difficulty_reward_tensors = [torch.tensor(r, dtype=torch.float32) for r in
                                      word_difficulty_rewards]
    return sent_len_reward_tensors, word_difficulty_reward_tensors


def get_max_new_tokens(step: int,
                       start: int = 50,
                       end: int = 240,
                       curriculum_steps: int = 80) -> int:
    """
    Calculates the `max_new_tokens` for the current training step based on a linear
    curriculum.

    Args:
        step: Current training step.
        start: Initial `max_new_tokens` value.
        end: Final `max_new_tokens` value.
        curriculum_steps: Total number of steps when max_new_tokens plateaus.

    Returns:
        int: The calculated `max_new_tokens` for the current step.
    """
    if step >= curriculum_steps:
        return end
    return int(start + (end - start) * (step / curriculum_steps))


if __name__ == "__main__":
    torch.manual_seed(SEED + 6103)

    parser = argparse.ArgumentParser(description="Rewriting complex scholarly abstracts to laymen.")
    # fmt: off
    # ppo relevant
    parser.add_argument("--task_name", type=str, default=PROJECT_NAME, help="Experiment name for tracking")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate for optimizer")
    parser.add_argument("--steps", type=int, default=20000, help="Number of training steps")
    parser.add_argument("--mini_batch_size", type=int, default=2, help="Mini batch size for PPO updates")
    parser.add_argument("--ppo_epochs", type=int, default=2, help="Number of optimization rollouts per batch of samples during PPO training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--early_stopping", type=lambda x: (str(x).lower() == 'true'), default=False, help="Enable early stopping if KL divergence is too high")
    parser.add_argument("--target_kl", type=float, default=1.0, help="Target KL divergence for early stopping")
    parser.add_argument("--use_score_scaling", action="store_true", help="Enable score scaling")
    parser.add_argument("--use_score_norm", action="store_true", help="Enable score normalization")
    parser.add_argument("--score_clip", type=float, default=None, help="Value to clip the scores, use 'None' to disable")
    # misc
    parser.add_argument("--eval_interval", type=int, default=10, help="Interval between evaluations")
    parser.add_argument("--num_eval_samples", type=int, default=32, help="Num of samples for evaluation")
    parser.add_argument("--save_folder", type=str, default=f"ppo_{CLM_MODEL_NAME}", help="Experiment name for checkpointing, under the directory" "of ckpts")
    parser.add_argument("--normalize_reward", type=bool, default=False, help="Normalize rewards in z score")
    parser.add_argument("--sft_ckpt_path", type=str, help="Path to the SFT'ed model")
    parser.add_argument("--max_new_tokens", type=int, default=300, help="Max rollout length")
    # curriculum for the rollouts
    parser.add_argument("--enable_curriculum", type=lambda x: (str(x).lower() == 'true'), default=False, help="Enable curriculum learning for rollout length")
    parser.add_argument("--rollout_curriculum", nargs=3, type=int, default=[50, 240, 60], help="Tuple indicating the start, end, and steps for rollout curriculum")
    # fmt: on
    args = parser.parse_args()
    # ignore the extra args that are not for ppo
    config_kwargs = vars(args).copy()
    keys_to_pop = [
        "eval_interval",
        "num_eval_samples",
        "save_folder",
        "normalize_reward",
        "sft_ckpt_path",
        "max_new_tokens",
        "enable_curriculum",
        "rollout_curriculum"
    ]
    for key in keys_to_pop:
        config_kwargs.pop(key, None)
    # config ppo
    learning_rate = 7.07e-6
    adap_kl_ctrl = True
    init_kl_coef = 0.2  # double check
    gamma = 1.0  # pretty sure
    lam = todo
    kl_penalty = 'kl'
    target = 6
    horizon = 10000
    # no dropout



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
        "min_length": 10,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        # fixed rollout length
        "max_new_tokens": args.max_new_tokens,
    }

    for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        # rollout curriculum
        if args.enable_curriculum:
            _max_new_tokens = get_max_new_tokens(step,
                                                 start=args.rollout_curriculum[0],
                                                 end=args.rollout_curriculum[1],
                                                 curriculum_steps=args.rollout_curriculum[2])
        else:
            _max_new_tokens = args.max_new_tokens
        rollout_kwargs["max_new_tokens"] = _max_new_tokens

        query_tensors = batch["input_ids"]

        response_tensors, ref_response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            generate_ref_response=True,
            **rollout_kwargs,
        )

        batch["response"] = tokenizer.batch_decode(response_tensors)
        batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

        rewards = reward2tensor(batch["response"], compute_ari, args.normalize_reward)
        # ref rewards
        ref_rewards = reward2tensor(
            batch["ref_response"], compute_ari, args.normalize_reward
        )
        batch["ref_rewards"] = ref_rewards
        batch["advantage"] = [p - r for p, r in zip(rewards, ref_rewards)]

        # execute a PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(
            stats,
            batch,
            rewards,
            columns_to_log=[
                "query",
                "response",
                "ref_response",
                "ref_rewards",
                "advantage",
            ],
        )
        wandb.log(rollout_kwargs)

        # evaluate on validation set after every n steps
        if step % args.eval_interval == 0:
            eval_score = evaluate_model(
                model=policy_model,
                dataset=dataset["validation"],
                tokenizer=tokenizer,
                compute_ari_func=compute_ari,
                num_samples=args.num_eval_samples,
            )
            wandb.log(eval_score)
            print(f"Step: {step}, Eval ARI: {np.mean(eval_score['ari']):.2f}")
            print(f"Step: {step}, Eval BLEU: {np.mean(eval_score['sacrebleu']):.2f}")

            # save top-3 checkpoints and the last one
            save_checkpoint(
                model=policy_model,
                step=step,
                eval_score=eval_score,
                save_folder=args.save_folder,
            )
