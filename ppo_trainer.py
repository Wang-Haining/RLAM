import argparse
import gc
import heapq
import json
import os
import pickle
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union, Any

import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from nltk.tokenize import sent_tokenize
from sacremoses import MosesTokenizer
from torch.utils.data import DataLoader
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification,
                          DataCollatorWithPadding, GenerationConfig,
                          PreTrainedTokenizer, Trainer, TrainerCallback,
                          TrainerControl, TrainerState, TrainingArguments)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer_callback import CallbackHandler, DefaultFlowCallback

from ppo_utils import (disable_dropout_in_model, exact_div, first_true_indices,
                       forward, generate, get_reward, masked_mean,
                       masked_whiten, prepare_deepspeed, print_rich_table,
                       truncate_response, unwrap_model_for_generation)
from utils import (CKPTS_DIR, MAX_NEW_TOKENS, PROJECT_NAME, SEED, SEP_TOKENS,
                   TASK_PREFIX, WORD_ACCESSIBILITY_MODEL, WORD_FREQ_CSV,
                   build_dataset, collator, compute_ari, compute_sent_len,
                   compute_token_accessibility, read_token_frequencies)

# from trl import (AutoModelForCausalLMWithValueHead,
#                  AutoModelForSeq2SeqLMWithValueHead,
#                  PPOConfig, PPOTrainer, set_seed)


nltk.download('punkt')
INVALID_LOGPROB = 1.0


# get word frequencies and the model to predict rare words' accessibility
token_freq = read_token_frequencies(WORD_FREQ_CSV)
top_100k_tokens = heapq.nlargest(100000, token_freq, key=token_freq.get)
# load for making predictions word accessibility
wa_model = pickle.load(open(WORD_ACCESSIBILITY_MODEL, 'rb'))
total_tokens = sum(token_freq.values())
current_device = Accelerator().local_process_index


def compute_uam_score(responses: List[str],
                        top_100k_tokens=top_100k_tokens,
                        wa_model=wa_model,
                        total_tokens=total_tokens,
                        token_freq=token_freq) -> Dict[str, List[torch.Tensor]]:
    """
    Calculate rewards for a batch of responses:
    - Avg sentence length: Computed over all sentences in a response. (Note, because we
        want average to be shorter, we need to negate sentence length in order to
        maximize the rewards.)
    - Avg word accessibility: Defined as the negative logarithm of the token frequency
      per billion, based on its occurrences in the English Wikipedia corpus.
    - Sentence count reward: Penalizes deviation from the target number of sentences.

    During pilot running, models cheat by spitting out only one eos token. So we
    penalize both word accessibility and sentence length with reasonably large negative
    feedback in such situations.

    Note, we intentionally preclude the calculation of EOS tokens as their inclusion
    will lead to underestimated word accessibility and inflated sentence length.

    Parameters:
        responses: A list of response strings to process.
        top_100k_tokens: Set of the top 100k tokens based on frequency.
        wa_model: The model used to estimate word accessibility.
        total_tokens: Total number of tokens in the reference corpus.
        token_freq: Dictionary of token frequencies.

    Returns:
        A dict containing three lists of tensors:
        - Raw sentence length rewards.
        - Raw word accessibility rewards.
    """
    sent_len_rewards = []
    word_accessibility_rewards = []
    mt = MosesTokenizer(lang='en')

    for i, response in enumerate(responses):
        # penalize too short generations
        if len(response.strip()) <= 50:
            sent_len_rewards.append(25.0)
            word_accessibility_rewards.append(7.0)
        else:
            sent_len_list = []
            word_accessibility_list = []
            sents = sent_tokenize(response)

            for sent in sents:
                # prevent noise from artificial eos tokens
                for t in SEP_TOKENS:
                    if sent.strip().endswith(t) or sent.strip().startswith(t):
                        sent = sent.replace(t, "").strip()
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

    # negate sentence length and count for intuitive reward maximization
    sent_len_rewards = torch.stack([-1.0 * torch.tensor(r, dtype=torch.float32) for r in sent_len_rewards])
    word_accessibility_rewards = torch.stack([torch.tensor(r, dtype=torch.float32) for r in word_accessibility_rewards])

    return {"sl_score": sent_len_rewards, "wa_score": word_accessibility_rewards}


@dataclass
class OnpolicyRuntimeConfig:
    # various batch sizes
    world_size: Optional[int] = None
    """The number of processes (GPUs) to use"""
    num_updates: Optional[int] = None
    """The number of updates to train"""
    micro_batch_size: Optional[int] = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_batch_size: Optional[int] = None
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""
    batch_size: Optional[int] = None
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    local_mini_batch_size: Optional[int] = None
    """the mini batch size per GPU"""
    mini_batch_size: Optional[int] = None
    """the mini batch size across GPUs"""

@dataclass
class PPOConfig(TrainingArguments):
    # see https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.TrainingArguments
    # for common arguments controlling saving/checkpointing/logging/eval
    # common config
    exp_name: str = 'RLUAM'
    """the name of this experiment"""
    run_name: Optional[str] = None
    """a unique name of this run"""
    sanity_check: bool = False
    """whether to run in debug mode"""

    # batch size related config
    num_mini_batches: int = 1
    """Number of minibatches to split a batch into"""
    total_episodes: Optional[int] = None
    """The total number of episodes in the dataset"""
    local_rollout_forward_batch_size: int = 64
    """per rank no grad forward pass in the rollout phase"""
    num_sample_generations: int = 10
    """the number of debugging samples generations (i.e., `generate_completions` calls) throughout training"""

    # other config
    base_model: str = "google/gemma-2b"
    """the name of the pretrained model to use"""
    response_length: int = 256
    """the length of the response"""
    stop_token: Optional[Literal["eos"]] = None  # <eos> for gemma
    """the stop token"""
    stop_token_id: Optional[int] = None  # 1 for gemma <eos>
    """the truncation token id"""
    temperature: float = 0.7
    """the sampling temperature"""
    # fixme: to rm
    # penalty_reward_value: int = -1
    # """the reward value for responses that do not contain `stop_token_id`"""
    # non_eos_penalty: bool = False
    # """whether to penalize responses that do not contain `stop_token_id`"""
    # reward_model_path: str = None
    # """the path to the reward model"""
    sft_model_path: str = None
    """the path to the sft model"""

    # ppo config
    num_ppo_epochs: int = 4
    """the number of epochs to train"""
    vf_coef: float = 0.1
    """the value function coefficient"""
    cliprange: float = 0.2
    """the clip range"""
    cliprange_value: float = 0.2
    """the clip range for the value function"""
    gamma: float = 1
    """the discount factor"""
    lam: float = 0.95
    """the lambda value for GAE"""
    whiten_rewards: bool = False
    """whether to whiten the rewards"""
    # todo: perhaps with a KL controller
    kl_coef: float = 0.05
    """the KL coefficient"""

    # uncombined accessibility measure related
    sl_coef: float = 1.0
    "Scaling factor for sentence length reward (will keep this frozen as 1.0)"
    wa_coef: float = 2.0
    "Scaling factor for word accessibility reward (will vary for an optimal value)"

    # logging and evaluation intervals (directly inherited from TrainingArguments)
    logging_steps: int = 2
    save_steps: int = 10
    eval_steps: int = 10
    evaluation_strategy: Optional[str] = "steps"  # "no", "steps", "epoch"
    save_strategy: Optional[str] = "steps"  # "no", "epoch", "steps"
    save_total_limit: Optional[int] = 3
    output_dir: str = 'ckpts'
    overwrite_output_dir: bool = False


def parse_args() -> Any:
    parser = argparse.ArgumentParser(description="Training script for PPO")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()
    return args

def load_config(json_file: str) -> Any:
    with open(json_file, "r") as file:
        config_dict = json.load(file)
    return PPOConfig(**config_dict)


class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, value_model) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model
        self.critic_backbone = getattr(value_model, value_model.base_model_prefix)

    def forward(self, **kwargs):
        output = self.critic_backbone(
            **kwargs,
        )
        logits = self.value_model.score(output.hidden_states[-1])
        return (self.policy(**kwargs),  # batch_size, seq_length, vocab_size
                logits)  # batch_size, seq_length


class PPOTrainer(Trainer):
    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        policy: nn.Module,
        ref_policy: nn.Module,
        train_dataset: Dataset,
        value_model: Optional[nn.Module] = None,
        # data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[List[TrainerCallback]] = None,
    ) -> None:
        self.args = config
        args = config
        self.tokenizer = tokenizer
        self.policy = policy
        # will output fixed-length sequences and tokens generated after an eos token will be ignored anyway
        self.policy.generation_config.eos_token_id = None  # disable `pad_token_id` and `eos_token_id` because we just want to
        self.policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding

        self.ref_policy = ref_policy
        # fixme: to rm
        # self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.value_model = value_model
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers
        self.callbacks = callbacks

        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = args.num_train_epochs * self.train_dataset_len
        self.accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        args.world_size = self.accelerator.num_processes
        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        if args.whiten_rewards:
            assert (
                args.local_mini_batch_size >= 8
            ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        args.num_updates = args.total_episodes // args.batch_size
        time_tensor = torch.tensor(int(time.time()), device=self.accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + self.accelerator.process_index * 100937  # a large prime number
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_updates // args.num_sample_generations)

        #########
        # setup model, optimizer, and others
        #########
        # fixme: removed reward_model below
        for module in [policy, ref_policy, value_model]:
            disable_dropout_in_model(module)
        if args.stop_token == "eos":
            args.stop_token_id = tokenizer.eos_token_id
        self.model = PolicyAndValueWrapper(policy, value_model)
        self.create_optimizer_and_scheduler(num_training_steps=args.num_updates)

        #########
        ### trainer specifics
        #########
        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )
        DEFAULT_CALLBACKS = [DefaultFlowCallback]
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        if self.callbacks is None:
            self.callbacks = default_callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.control = TrainerControl()
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.backup_model = None

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=args.local_batch_size,
            shuffle=True,
            collate_fn=DataCollatorWithPadding(tokenizer),
            drop_last=True,
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(self.model, self.optimizer, self.dataloader)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=DataCollatorWithPadding(self.tokenizer),
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)

        if self.is_deepspeed_enabled:
            # fixme: to rm
            # self.reward_model = prepare_deepspeed(self.reward_model, args.per_device_train_batch_size)
            self.ref_policy = prepare_deepspeed(self.ref_policy, args.per_device_train_batch_size)
        else:
            # fixme: to rm
            # self.reward_model = self.reward_model.to(self.accelerator.device)
            self.ref_policy = self.ref_policy.to(self.accelerator.device)

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader


    def push_to_hub(self, **kwargs):
        """Modified from `Trainer.save_model` to only save the policy and not the value network."""
        self.backup_model = self.model
        self.model = self.accelerator.unwrap_model(self.model).policy  # save only the policy
        super().push_to_hub(**kwargs)
        self.model = self.backup_model


    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Modified from `Trainer.save_model` to only save the policy and not the value network."""
        if not _internal_call:  # `push_to_hub` already swaps out the self.model with policy
            self.backup_model = self.model
            self.model = self.accelerator.unwrap_model(self.model).policy  # save only the policy
        if output_dir is None:
            output_dir = self.args.output_dir
        state_dict = self.accelerator.get_state_dict(self.backup_model)
        policy_state_dict = state_dict
        if self.accelerator.is_main_process:
            policy_state_dict = OrderedDict(
                {k[len("policy."):]: v for k, v in state_dict.items() if k.startswith("policy.")}
            )
        if self.args.should_save:
            self._save(output_dir, state_dict=policy_state_dict)
        if not _internal_call:
            self.model = self.backup_model

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_policy
        # fixme: to rm
        # reward_model = self.reward_model
        tokenizer = self.tokenizer
        dataloader = self.dataloader
        device = self.accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            min_new_tokens=args.response_length,
            # fixme: to rm
            # min_new_tokens=-1 if 'flan' not in args.base_model.lower() else 20,
            temperature=(args.temperature + 1e-8),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        accelerator.print("===training policy===")
        global_step = 0
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # 1. rollout phase:
        for update in range(1, args.num_updates + 1):
            global_step += 1 * args.batch_size
            self.lr_scheduler.step()
            data = next(iter_dataloader)
            with torch.no_grad():
                queries = data["input_ids"].to(device)
                context_length = queries.shape[1]
                query_responses = []
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                values = []
                scores = []
                sequence_lengths = []
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                        query = queries[i: i + args.local_rollout_forward_batch_size]
                        query_response, logits = generate(
                            unwrapped_model.policy,
                            query,
                            tokenizer.pad_token_id,
                            generation_config,
                        )  # logits of shape local_rollout_forward_batch_size, seq_len, vocab_size
                        response = query_response[:, context_length:]  # local_rollout_forward_batch_size, seq_len - context_len

                        # use the logits during generation directly, instead of using the following
                        all_logprob = F.log_softmax(logits, dim=-1)  # local_rollout_forward_batch_size, seq_len, vocab_size
                        # log-probabilities of the generated tokens for each sequence in the batch.
                        logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)  # local_rollout_forward_batch_size, seq_len - context_length
                        del logits, all_logprob
                        torch.cuda.empty_cache()

                        ref_output = forward(ref_policy, query_response, tokenizer.pad_token_id)
                        ref_logits = ref_output.logits[:, context_length - 1: -1]
                        ref_logits /= args.temperature + 1e-8
                        ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                        ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)  # local_rollout_forward_batch_size, seq_len - context_length
                        del ref_output, ref_logits, ref_all_logprob
                        torch.cuda.empty_cache()

                        # fixme: reward func: score
                        # Response Processing 1. truncate response after the first occurrence of `stop_token_id` and
                        # pad up to the maximum sequence length within the batch
                        postprocessed_response = truncate_response(
                            args.stop_token_id, tokenizer.pad_token_id, response
                        )

                        # Response Processing 2. run reward model on the truncated responses
                        # postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                        sequence_length = first_true_indices(postprocessed_response == tokenizer.pad_token_id) - 1  # batch_size,
                        unwrapped_value_model = accelerator.unwrap_model(model).value_model
                        # get value estimates for generated tokens, i.e., `value`
                        full_value, _, _ = get_reward(
                            unwrapped_value_model, query_response, tokenizer.pad_token_id, context_length
                        )
                        value = full_value[:, context_length - 1: -1].squeeze(-1)
                        # fixme: use funcs not model
                        generated_texts = tokenizer.batch_decode(postprocessed_response,
                                                                 skip_special_tokens=True,
                                                                 clean_up_tokenization_spaces=True)
                        uam_score = compute_uam_score(generated_texts)
                        score = self.args.sl_coef * uam_score['sl_score'] + self.args.wa_coef * uam_score['wa_score']
                        score = score.to(device=self.accelerator.device)

                        query_responses.append(query_response)
                        responses.append(response)
                        postprocessed_responses.append(postprocessed_response)
                        logprobs.append(logprob)
                        ref_logprobs.append(ref_logprob)
                        values.append(value)
                        sequence_lengths.append(sequence_length)
                        scores.append(score)
                query_responses = torch.cat(query_responses, 0)
                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                values = torch.cat(values, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                del (logprob, ref_logprob, full_value, value, score, unwrapped_model)
                torch.cuda.empty_cache()
                gc.collect()

                # fixme: we don't care about whether eos tokens are generated
                # Response Processing 3. filter response. Ensure that the sample contains stop_token_id
                # responses not passing that filter will receive a low (fixed) score
                # only query humans on responses that pass that filter
                # contain_eos_token = torch.any(postprocessed_responses == tokenizer.eos_token_id, dim=-1)
                # if args.non_eos_penalty:
                #     scores = torch.where(contain_eos_token, scores, torch.full_like(scores, args.penalty_reward_value))
                # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")


                # be very careful with `padding_mask_p1`
                # see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                # values are computed for each token in the entire sequence including the previouly generated tokesn
                # whereas logprobs are computed only for the generated tokens
                sequence_lengths_p1 = sequence_lengths + 1
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
                values = torch.masked_fill(values, padding_mask_p1, 0)

                # 4. compute rewards
                kl = logprobs - ref_logprobs  # batch_size, seq_len - context_length
                non_score_reward = -args.kl_coef * kl  # batch_size, seq_len - context_length
                rewards = non_score_reward.clone()  # batch_size, seq_len - context_length
                actual_start = torch.arange(rewards.size(0), device=rewards.device)  # batch_size,
                actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)  # batch_size,
                # add reward (from reward func) to the last position
                rewards[[actual_start, actual_end]] += scores  # batch_size, seq_len - context_length

                # fixme: remove this option
                # 5. whiten rewards
                if args.whiten_rewards:
                    rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
                    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

                # 6. compute advantages and returns
                lastgaelam = 0
                advantages_reversed = []
                gen_length = responses.shape[1]
                for t in reversed(range(gen_length)):
                    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = advantages + values
                advantages = masked_whiten(advantages, ~padding_mask)  # batch_size, gen_length
                advantages = torch.masked_fill(advantages, padding_mask, 0)
                torch.cuda.empty_cache()

            # ppo training: do multiple epochs with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                            mb_return = returns[micro_batch_inds]
                            mb_advantage = advantages[micro_batch_inds]
                            mb_values = values[micro_batch_inds]
                            mb_responses = responses[micro_batch_inds]
                            mb_query_responses = query_responses[micro_batch_inds]
                            mb_logprobs = logprobs[micro_batch_inds]

                            output, vpred_temp = forward(model, mb_query_responses, tokenizer.pad_token_id)
                            logits = output.logits[:, context_length - 1: -1]
                            logits /= args.temperature + 1e-8
                            new_all_logprobs = F.log_softmax(logits, dim=-1)
                            new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                            new_logprobs = torch.masked_fill(
                                new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                            )
                            vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                            vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds], 0)
                            vpredclipped = torch.clamp(
                                vpred,
                                mb_values - args.cliprange_value,
                                mb_values + args.cliprange_value,
                            )
                            vf_losses1 = torch.square(vpred - mb_return)
                            vf_losses2 = torch.square(vpredclipped - mb_return)
                            vf_loss_max = torch.max(vf_losses1, vf_losses2)
                            vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1[micro_batch_inds])
                            vf_clipfrac = masked_mean(
                                (vf_losses2 > vf_losses1).float(), ~padding_mask_p1[micro_batch_inds]
                            )
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
                            loss = pg_loss + args.vf_coef * vf_loss
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            # calculating statistics
                            with torch.no_grad():
                                pg_clipfrac = masked_mean(
                                    (pg_losses2 > pg_losses).float(), ~padding_mask[micro_batch_inds]
                                )
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[
                                    ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx
                                ] = pg_clipfrac
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                                vf_clipfrac_stats[
                                    ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx
                                ] = vf_clipfrac
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, vpred_temp, logits, new_all_logprobs, new_logprobs, vpred, vpredclipped,
                        vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max,
                        pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,
                        mb_advantage, mb_values, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.sum(1).mean()
                rlhf_reward = mean_non_score_reward + scores.mean()
                eps = int(global_step / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = self.accelerator.gather(mean_non_score_reward).mean().item()
                metrics["objective/rlhf_reward"] = self.accelerator.gather(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather(scores.mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather(pg_loss_stats).mean().item()
                metrics["loss/value_avg"] = self.accelerator.gather(vf_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather(vf_clipfrac_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == tokenizer.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = global_step
                self.state.epoch = global_step / self.train_dataset_len  # used by self.log
                self.log(metrics)
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores, metrics, non_score_reward
            torch.cuda.empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)
                torch.cuda.empty_cache()
            del (
                query_responses,
                responses,
                postprocessed_responses,
                logprobs,
                ref_logprobs,
                values,
                sequence_lengths,
                # contain_eos_token,
                sequence_lengths_p1,
                response_idxs,
                padding_mask,
                padding_mask_p1,
                rewards,
                actual_start,
                actual_end,
                advantages,
                returns,
            )
            torch.cuda.empty_cache()

    def generate_completions(self, sampling: bool = False):
        args = self.args
        tokenizer = self.tokenizer
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.01 + 1e-8),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        table = defaultdict(list)
        for batch in self.eval_dataloader:
            query = batch["input_ids"]
            with torch.no_grad():
                context_length = query.shape[1]
                with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                    query_response, _ = generate(
                        unwrapped_model.policy,
                        query,
                        tokenizer.pad_token_id,
                        generation_config,
                    )
                response = query_response[:, context_length:]
                postprocessed_response = response
                if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                    postprocessed_response = truncate_response(args.stop_token_id, tokenizer.pad_token_id, response)
                table["query"].extend(gather_object(tokenizer.batch_decode(query, skip_special_tokens=True, clean_up_tokenization_spaces=True)))
                table["model response"].extend(gather_object(tokenizer.batch_decode(postprocessed_response, skip_special_tokens=True, clean_up_tokenization_spaces=True)))

                # postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                generated_texts = tokenizer.batch_decode(postprocessed_response,
                                                         skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=True)
                uam_score = compute_uam_score(generated_texts)
                score = self.args.sl_coef * uam_score['sl_score'] + self.args.wa_coef * uam_score['wa_score']
                score = score.to(device=self.accelerator.device)
                table["score"].extend(self.accelerator.gather(score).float().cpu().numpy())

            if sampling:
                break
        df = pd.DataFrame(table)
        if self.accelerator.process_index == 0:
            print_rich_table(df.iloc[0: 0 + 5])
        if "wandb" in args.report_to:
            import wandb

            if wandb.run is not None:
                wandb.log({"completions": wandb.Table(dataframe=df)})


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config_path)

    # build dataset
    dataset = build_dataset(model_name=config.sft_model_path,
                            task_prefix=TASK_PREFIX)
    train_dataset = dataset['train']
    eval_dataset = dataset['validation']

    # initialize tokenizer and models
    tokenizer = PreTrainedTokenizer.from_pretrained(config.base_model)
    value_model = AutoModelForSequenceClassification.from_pretrained(config.sft_model_path, num_labels=1, torch_dtype=torch.bfloat16)
    ref_policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path, torch_dtype=torch.bfloat16)
    policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path, torch_dtype=torch.bfloat16)

    # training Loop
    ppo_trainer = PPOTrainer(config=config,
                             tokenizer=tokenizer,
                             policy=policy,
                             ref_policy=ref_policy,
                             train_dataset=train_dataset,
                             value_model=value_model,
                             # data_collator=DataCollatorWithPadding,
                             eval_dataset=eval_dataset)
    ppo_trainer.train()
