"""
The code is modified from
https://github.com/vwxyzjn/summarize_from_feedback_details/blob/main/summarize_from_feedback_details/ppo.py
"""

import heapq
import os
import pickle
import random
import shutil
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Literal, Optional, Tuple

import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import wandb
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset, load_dataset
from nltk.tokenize import sent_tokenize
from rich.console import Console
from rich.pretty import pprint
from sacrebleu.metrics import BLEU
from sacremoses import MosesTokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoTokenizer, GenerationConfig, PretrainedConfig,
                          PreTrainedModel, PreTrainedTokenizerBase)

from utils import (SEP_TOKENS, WORD_ACCESSIBILITY_MODEL, WORD_FREQ_CSV,
                   build_ppo_dataset, compute_ari, compute_sent_len,
                   compute_token_accessibility, count_sent,
                   read_token_frequencies)

torch.set_printoptions(precision=3, sci_mode=False)
bleu = BLEU()
nltk.download('punkt')
INVALID_LOGPROB = 1.0

# init word accessibility predictor
# get word frequencies and the model to predict rare words' accessibility
token_freq = read_token_frequencies(WORD_FREQ_CSV)
top_100k_tokens = heapq.nlargest(100000, token_freq, key=token_freq.get)
# load for making predictions word accessibility
wa_model = pickle.load(open(WORD_ACCESSIBILITY_MODEL, 'rb'))
total_tokens = sum(token_freq.values())


def compute_uam_score(responses: List[str],
                      top_100k_tokens=top_100k_tokens,
                      wa_model=wa_model,
                      total_tokens=total_tokens,
                      token_freq=token_freq) -> Dict[str, torch.Tensor]:
    """
    Score a batch of responses:
    - Avg sentence length: Computed over all sentences in a response. (Note, because we
        want average to be shorter, we need to negate sentence length in order to
        maximize the score.)
    - Avg word accessibility: Defined as the negative logarithm of the token frequency
      per billion, based on its occurrences in the English Wikipedia corpus.

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
        - Negative sentence length score.
        - Raw word accessibility score.
    """
    sent_len_rewards = []
    word_accessibility_rewards = []
    mt = MosesTokenizer(lang='en')
    for response in responses:
        # penalize too short generations
        if len(response.strip()) <= 50:
            sent_len_rewards.append(28.0)
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
    sent_len_rewards = torch.stack([-1.0 * torch.tensor(r, dtype=torch.float32) for r in sent_len_rewards])
    word_accessibility_rewards = torch.stack([torch.tensor(r, dtype=torch.float32) for r in word_accessibility_rewards])
    return {"sl_score": sent_len_rewards, "wa_score": word_accessibility_rewards}


@dataclass
class PpoHParams:
    nminibatches: int = 1
    noptepochs: int = 4
    vf_coef: float = 0.1
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    gamma: float = 1
    lam: float = 0.95
    whiten_rewards: bool = False
    # fixme: modify
    kl_coef: float = 0.2


@dataclass
class Args:
    # common args
    project_name: str = "RLUAM_TEST"
    # project_name: str = "Reinforcement Learning From Uncombined Accessibility Measures"
    """the name of this experiment"""
    run_name: Optional[str] = None
    """a unique name of this run"""
    seed: int = 42
    """seed of the experiment"""
    cuda: bool = True
    """Whether to use cuda if available."""
    # fixme: rm
    # deepspeed: bool = False
    # """Whether to use deepspeed to train the model"""
    print_sample_output_freq: int = 10
    """How often to print sample output"""
    run_eval: bool = True
    """Whether to run evaluation"""
    # todo: add an option to set num_eval_samples

    # optimizer (adamw) args
    eps: float = 1e-5
    """the epsilon value for adamw"""
    lr: float = 3e-6
    """the learning rate for adamw"""
    scheduler: str = "cosine"
    """Which scheduler to use"""
    warm_up_steps: int = 20
    """Number of warm up steps for the scheduler"""

    # various batch sizes
    world_size: Optional[int] = None
    """The number of processes (GPUs) to use"""
    num_train_epochs: int = 1
    """Number of epochs to train"""
    num_updates: Optional[int] = None
    """The number of updates to train"""
    gradient_accumulation_steps: int = 1
    """The number of gradient accumulation steps"""
    local_micro_batch_size: Optional[int] = 1
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    total_episodes: Optional[int] = 1000000
    """The total number of episodes in the dataset"""
    micro_batch_size: Optional[int] = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_batch_size: Optional[int] = None
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""
    batch_size: Optional[int] = None
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    nminibatches: int = 1
    """Number of minibatches to split a batch into"""
    local_mini_batch_size: Optional[int] = None
    """the mini batch size per GPU"""
    mini_batch_size: Optional[int] = None
    """the mini batch size across GPUs"""
    local_eval_batch_size: int = 1
    """per rank eval batch size"""
    local_rollout_forward_batch_size: int = 1
    """per rank no grad forward pass in the rollout phase"""

    # other args
    # base_model: str = "google/gemma-2b"
    base_model: str = 'allenai/OLMo-1B-hf'
    """the name of the pretrained model to use"""
    # fixme: modify
    # query_dataset: str = "vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144"
    # """the query dataset"""
    response_length: int = 256
    """the length of the response"""
    truncate_token: Literal["eos"] = "eos"
    """the truncate token"""
    # fixme: modify
    truncate_token_id: Optional[int] = 50279  # 1 for gemma, 50279 for olmo
    """the truncation token id"""
    temperature: float = 0.7
    """the sampling temperature"""
    penalty_reward_value: int = -5  # todo: tune
    """the reward value for responses that do not contain `truncate_token_id`"""
    non_eos_penalty: bool = True
    """whether to penalize responses that do not contain `truncate_token_id`"""
    # offload: bool = False
    # """Whether to offload ref policy and reward model to CPU"""  # fixme: reward model alarm
    # fixme: rm
    # reward_model_path: str = ""
    # """the path to the reward model"""
    # sft_model_path: str = "ckpts/sft_gemma-2b/checkpoint-1120"
    sft_model_path: str = "ckpts/sft_OLMo-1B-hf/checkpoint-1100"
    """the path to the sft model"""

    # reward related
    sl_coef: float = 1.0
    "Scaling factor for sentence length reward (will keep this frozen as 1.0)"
    wa_coef: float = 2.0
    "Scaling factor for word accessibility reward (will vary for an optimal value)"

    # logging and evaluation intervals (directly inherited from TrainingArguments)
    logging_steps: int = 2
    save_steps: int = 10
    eval_steps: int = 10
    num_eval_samples: int = 64
    # eval_strategy: Optional[str] = "steps"  # "no", "steps", "epoch"
    # save_strategy: Optional[str] = "steps"  # "no", "epoch", "steps"
    save_total_limit: Optional[int] = 3
    output_dir: str = 'ckpts/test_run'
    ppo: PpoHParams = field(default_factory=PpoHParams)
    """Default values will be used to create a PpoHParams"""


def parse_args() -> tuple[Args, Accelerator]:
    args = tyro.cli(Args)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    args.world_size = accelerator.num_processes
    args.local_batch_size = args.local_micro_batch_size * args.gradient_accumulation_steps * args.nminibatches
    args.micro_batch_size = int(args.local_micro_batch_size * args.world_size)
    args.batch_size = int(args.local_batch_size * args.world_size)
    args.mini_batch_size = exact_div(args.batch_size, args.nminibatches)
    args.local_mini_batch_size = exact_div(args.local_batch_size, args.nminibatches)
    if args.ppo.whiten_rewards:
        assert (
            args.local_mini_batch_size >= 8
        ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
    # `per_rank_rollout_batch_size` is our `args.local_batch_size`
    # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
    args.num_updates = args.total_episodes // args.batch_size
    time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
    time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
    args.run_name = f"{args.project_name}__{args.seed}__{time_int}"
    # if args.push_to_hub:
    #     if args.hf_repo_id is None: # auto-generate one
    #         args.hf_repo_id = f"{args.base_model.replace('/', '_')}__{args.exp_name}__tldr"
    #     if args.hf_entity is None:  # find the current user
    #         args.hf_entity = api.whoami()["name"]
    #     if "/" not in args.hf_repo_id: # prepend the current user
    #         args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
    #     if args.hf_repo_revision is None:  # auto-generate one
    #         args.hf_repo_revision = args.run_name
    #     args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"
    return args, accelerator


def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask, False)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


# fixme: rm
class ScalarModelConfig(PretrainedConfig):
    def __init__(
        self,
        base_model: str,
        base_config: PretrainedConfig = None,
        hidden_size: int = 768,
        bias: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        if base_config is None:
            base_config = AutoConfig.from_pretrained(base_model)
        self.base_config = base_config
        self.hidden_size = hidden_size
        self.bias = bias


class ScalarModel(PreTrainedModel):
    config_class = ScalarModelConfig
    def __init__(self, config: ScalarModelConfig):
        super().__init__(config)
        self.config = config
        self.lm_backbone = AutoModel.from_pretrained(
            config.base_model,
            config=self.config.base_config,
            trust_remote_code=True,
        )
        self.scalar_head = layer_init(
            nn.Linear(self.config.hidden_size, 1),
            std=1 / np.sqrt(self.config.hidden_size + 1),
        )

    def forward(self, **kwargs):
        output = self.lm_backbone(**kwargs)
        reward = self.scalar_head(output.hidden_states[-1]) - self.config.bias
        return reward


def get_reward(model, query_responses, tokenizer, context_length):
    attention_mask = query_responses != tokenizer.pad_token_id
    # position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    reward_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == tokenizer.pad_token_id) - 1 + context_length
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return (
        reward_logits,
        reward_logits[torch.arange(reward_logits.size(0), device=reward_logits.device), sequence_lengths].squeeze(-1),
        sequence_lengths,
    )


# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, critic) -> None:
        super().__init__()
        self.policy = policy
        self.critic = critic

    def forward(self, **kwargs):
        return self.policy(**kwargs), self.critic(**kwargs)


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError(f"Inexact division: {a} / {b} = {a / b}")
    return q


def generate(lm_backbone, queries, tokenizer, generation_config):
    """generate in a way that does not affect padding tokens"""
    context_length = queries.shape[1]
    attention_mask = queries != tokenizer.pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # generation collapsed if this was turned on. TODO: why does generation collapse with this?
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True
    )
    logits = torch.stack(output.scores, 1)
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1), logits


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of
    integers giving the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def truncate_response(args, tokenizer, responses):
    trunc_idxs = first_true_indices(responses == args.truncate_token_id).unsqueeze(-1)
    new_size = [1] * (len(responses.size()) - 1) + [responses.shape[1]]
    idxs = torch.arange(responses.shape[1], device=responses.device).view(*new_size)
    postprocessed_responses = torch.masked_fill(responses, idxs > trunc_idxs, tokenizer.pad_token_id)
    return postprocessed_responses


def forward(model, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    # position_ids = attention_mask.cumsum(1) - attention_mask.long()
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )


def evaluate_model(
        sl_coef: float,
        wa_coef: float,
        policy: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        dataloader: DataLoader,
        generation_config: GenerationConfig,
        num_samples: int
) -> Tuple[Dict[str, List], pd.DataFrame]:
    """
    Evaluates the policy model using various metrics on a subset of the dataset.

    Args:
        sl_coef (float): Scaling factor for sentence length score.
        wa_coef (float): Scaling factor for word accessibility score.
        policy (torch.nn.Module): The policy model to be evaluated.
        tokenizer (PreTrainedTokenizerBase): Tokenizer used for encoding/decoding.
        dataloader (DataLoader): DataLoader providing the evaluation dataset.
        generation_config (GenerationConfig): Configuration for text generation.
        num_samples (int): Number of samples to evaluate on.

    Returns:
        Tuple[Dict[str, List], pd.DataFrame]: Evaluation metrics and DataFrame of results.
    """
    eval_storage = defaultdict(list)
    bleu = BLEU()

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader)):
            # Evaluate reference response (i.e., human-written significance statement)
            reference_scores = compute_uam_score(data['response'])
            reference_sl_scores = reference_scores['sl_score']
            reference_wa_scores = reference_scores['wa_score']
            reference_total_scores = sl_coef * reference_sl_scores + wa_coef * reference_wa_scores

            # Evaluate policy generated response
            queries = data["query_token"]
            context_length = queries.shape[1]
            query_responses, _ = generate(
                policy,
                queries,
                tokenizer,
                generation_config,
            )
            responses = query_responses[:, context_length:]
            postprocessed_responses = truncate_response(args, tokenizer, responses)
            generated_texts = tokenizer.batch_decode(postprocessed_responses,
                                                     skip_special_tokens=True)

            # Calculate metrics
            uam_scores = compute_uam_score(generated_texts)
            sl_scores = uam_scores['sl_score']
            wa_scores = uam_scores['wa_score']
            total_scores = sl_coef * sl_scores + wa_coef * wa_scores
            ari_scores = []
            bleu_scores = []
            num_sents = []

            for g, r in zip(generated_texts, data['source']):
                ari_scores.append(compute_ari(g))
                bleu_scores.append(bleu.corpus_score([g], [[r]]).score)  # BLEU to the original abstract (cf. significance statement)
                num_sents.append(count_sent(g))

            eval_storage["queries"].extend(data['query'])  # str
            eval_storage["generated_texts"].extend(generated_texts)  # str
            eval_storage["total_scores"].append(total_scores.cpu().numpy().tolist())
            eval_storage["reference_responses"].extend(data['response'])  # str
            eval_storage["reference_scores"].append(
                reference_total_scores.cpu().numpy().tolist())
            eval_storage['ari'].extend(ari_scores)
            eval_storage['bleu'].extend(bleu_scores)
            eval_storage['sent_len'].extend(sl_scores.cpu().numpy().tolist())
            eval_storage['word_accessibility'].extend(wa_scores.cpu().numpy().tolist())
            eval_storage['sent_count'].extend(num_sents)

            if i >= num_samples:
                break

    eval_total_scores = [item for sublist in eval_storage["total_scores"] for item in
                         sublist]
    eval_reference_total_scores = [item for sublist in eval_storage["reference_scores"]
                                   for item in sublist]
    eval_df = pd.DataFrame(
        {
            "queries": gather_object(eval_storage["queries"]),
            "generated_texts": gather_object(eval_storage["generated_texts"]),
            "total_scores": gather_object(eval_total_scores),
            "reference_responses": gather_object(eval_storage["reference_responses"]),
            "reference_total_scores": gather_object(eval_reference_total_scores),
            "ari": eval_storage['ari'],
            "bleu": eval_storage['bleu'],
            "sent_len": eval_storage['sent_len'],
            "word_accessibility": eval_storage['word_accessibility'],
            "sent_count": eval_storage['sent_count']
        }
    )
    return eval_storage, eval_df


def save_model(accelerator, tokenizer, model, output_dir, ari, step, save_total_limit):
    save_path = os.path.join(output_dir, f"model_step_{step}_ari_{ari}")
    metadata_path = os.path.join(output_dir, "metadata.npz")

    # load existing metadata if available
    if os.path.exists(metadata_path):
        metadata = np.load(metadata_path, allow_pickle=True)
        saved_models = list(metadata['saved_models'])
    else:
        saved_models = []

    # save new model if conditions are met
    if len(saved_models) < save_total_limit or ari < max(m['ari'] for m in saved_models):
        # prepare model for saving
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)
            unwrapped = accelerator.unwrap_model(model).policy
            unwrapped.save_pretrained(save_path, save_function=accelerator.save)

        # update saved models list
        saved_models.append({
            'path': save_path,
            'ari': ari,
            'step': step
        })
        saved_models.sort(key=lambda x: x['ari'])

        # remove the worst model if limit exceeded
        if len(saved_models) > save_total_limit:
            worst_model = saved_models.pop(0)
            if os.path.exists(worst_model['path']):
                shutil.rmtree(worst_model['path'])

        # save updated metadata
        if accelerator.is_main_process:
            np.savez(metadata_path, saved_models=saved_models)


if __name__ == "__main__":
    args, accelerator = parse_args()
    local_seed = args.seed + accelerator.process_index * 100003  # a prime number

    # load dataset
    # dataset = load_dataset(args.query_dataset, split="train")
    dataset = build_ppo_dataset(args.base_model)
    dataset = dataset.with_format("torch", columns=["query_token", "reference_response_token", "query", 'response', 'source'])  # query_token: (bs, 512) left padded
    dataloader = DataLoader(dataset['train'], batch_size=args.local_batch_size, shuffle=True)
    eval_dataloaders = {}
    # fixme
    for split in ["validation"]:  # fixme: no test for now
        eval_dataset = dataset[split]
        eval_dataloaders[split] = DataLoader(eval_dataset, batch_size=args.local_eval_batch_size)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
        trust_remote_code=True,
    )
    # fixme: use <pad> is fine?
    # we use the padding token manually but do not resize the token embedding of the model
    # tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if args.truncate_token == "eos":
        args.truncate_token_id = tokenizer.eos_token_id

    console = Console(force_terminal=True)
    writer = SimpleNamespace()  # dummy writer
    writer.add_scalar = lambda x, y, z: None
    writer.add_histogram = lambda x, y, z: None
    if accelerator.is_main_process:
        wandb.init(
            project=args.project_name,
            sync_tensorboard=True,
            config=asdict(args),
            name=args.run_name,
            save_code=True,
        )
        file_extensions = [".toml", ".lock", ".py", ".sh", ".yaml"]
        wandb.run.log_code(".", include_fn=lambda path: any([path.endswith(ext) for ext in file_extensions]))
        writer = SummaryWriter(f"logs/{args.run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        pprint(args)
    device = accelerator.device
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.backends.cudnn.deterministic = True
    # init value model from scratch but ref and policy model from sft ckpt
    model_config = AutoConfig.from_pretrained(args.base_model)
    scalar_model_config = ScalarModelConfig(
        base_model=args.base_model,
        base_config=model_config,
        hidden_size=model_config.hidden_size,
    )

    critic = ScalarModel(scalar_model_config)
    ref_policy = AutoModelForCausalLM.from_pretrained(args.sft_model_path, config=model_config, torch_dtype=torch.bfloat16, trust_remote_code=True)
    policy = AutoModelForCausalLM.from_pretrained(args.sft_model_path, config=model_config, torch_dtype=torch.bfloat16, trust_remote_code=True)
    for module in [policy, ref_policy, critic]:
        disable_dropout(module)
    # will output fixed-length sequences and tokens generated after an eos token will be ignored anyway
    policy.generation_config.eos_token_id = None  # disable `pad_token_id` and `eos_token_id` because we just want to
    policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    # wrap policy and value head together
    model = PolicyAndValueWrapper(policy, critic)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps)

    # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
    # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
    torch.manual_seed(args.seed)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    eval_dataloaders = {split: accelerator.prepare(eval_dataloader) for split, eval_dataloader in eval_dataloaders.items()}
    torch.manual_seed(local_seed)  # reset the local seed again

    def repeat_generator():
        while True:
            yield from dataloader

    iter_dataloader = iter(repeat_generator())
    # fixme: rm
    # if args.deepspeed:
    #     import deepspeed
    #
    #     deepspeed_states = AcceleratorState().deepspeed_plugin
    #     deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"] = args.local_micro_batch_size
    #
    #     eval_ds_config = {
    #         "train_micro_batch_size_per_gpu": deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"],
    #         "bf16": {"enabled": True},
    #         "prescale_gradients": False,
    #         "wall_clock_breakdown": False,
    #     }
    #     if args.offload or args.base_model == "EleutherAI/pythia-6.9b-deduped":
    #         deepspeed_states.deepspeed_config["checkpoint"] = {"use_node_local_storage": True}
    #         eval_ds_config["zero_optimization"] = {
    #             "stage": 3,
    #             "stage3_param_persistence_threshold": 1e4,
    #             "offload_param": {"device": "cpu"},
    #         }
    #     accelerator.print(f"{eval_ds_config=}")
    #     reward_model, *_ = deepspeed.initialize(model=reward_model, config=eval_ds_config)
    #     reward_model.eval()
    #     ref_policy, *_ = deepspeed.initialize(model=ref_policy, config=eval_ds_config)
    #     ref_policy.eval()
    # else:
    ref_policy = ref_policy.to(device)
    # reward_model = reward_model.to(device)

    generation_config = GenerationConfig(
        max_new_tokens=args.response_length,
        min_new_tokens=args.response_length,
        temperature=(args.temperature + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )
    # use the same `0.01` temperature for validation response generation
    # https://github.com/openai/summarize-from-feedback/blob/700967448d10004279f138666442bf1497d0e705/exps/sample.py#L27
    validation_generation_config = GenerationConfig(
        max_new_tokens=args.response_length,
        min_new_tokens=args.response_length,
        temperature=(0.01 + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )

    accelerator.print("===training policy===")
    global_step = 0
    start_time = time.time()
    eval_split = list(eval_dataloaders.keys())[0]
    stats_shape = (args.ppo.noptepochs, args.nminibatches, args.gradient_accumulation_steps)
    approxkl_stats = torch.zeros(stats_shape, device=device)
    pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
    pg_loss_stats = torch.zeros(stats_shape, device=device)
    vf_loss_stats = torch.zeros(stats_shape, device=device)
    vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
    entropy_stats = torch.zeros(stats_shape, device=device)
    ratio_stats = torch.zeros(stats_shape, device=device)
    model.train()
    for update in range(1, args.num_updates + 1):
        global_step += 1 * args.batch_size
        if global_step < args.warm_up_steps:
            # linear warmup
            lr_scale = global_step / args.warm_up_steps
        else:
            # post warmup: decay the learning rate
            frac = 1.0 - (update - 1.0) / args.num_updates
            lr_scale = frac
        # global_step += 1 * args.batch_size
        # frac = 1.0 - (update - 1.0) / args.num_updates
        lrnow = lr_scale * args.lr
        optimizer.param_groups[0]["lr"] = lrnow
        data = next(iter_dataloader)
        with torch.no_grad():
            eval_storage, eval_df = evaluate_model(
                args.sl_coef, args.wa_coef,
                accelerator.unwrap_model(model).policy,
                tokenizer,
                eval_dataloaders[eval_split],
                validation_generation_config,
                num_samples=4  # test a few samples to see if this works
            )
            validation_score = np.mean(eval_storage["total_scores"])
            if args.print_sample_output_freq > 0 and (update - 1) % args.print_sample_output_freq == 0:
                if accelerator.is_main_process:
                    # fixme: no need to save dfs
                    eval_ds = Dataset.from_pandas(eval_df)
                    # eval_ds.save_to_disk(f"runs/{args.run_name}/{eval_split}_dataset_{global_step}")
                    # fixme: rm
                    # if args.track:
                    wandb.log({f"sft_samples/{eval_split}_query_responses": wandb.Table(dataframe=eval_df)}, step=update)
            del eval_storage, eval_df
            torch.cuda.empty_cache()
            # rollout phase
            queries = data["query_token"].to(device)
            context_length = queries.shape[1]
            query_responses = []
            responses = []
            postprocessed_responses = []
            logprobs = []
            ref_logprobs = []
            values = []
            scores = []
            sequence_lengths = []
            for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                query = queries[i: i + args.local_rollout_forward_batch_size]
                query_response, logits = generate(
                    accelerator.unwrap_model(model).policy,
                    query,
                    tokenizer,
                    generation_config,
                )
                response = query_response[:, context_length:]  # (local_rollout_forward_batch_size, gen_len)
                # print(f'rollout: {response}')
                # use the logits during generation directly, instead of using the following
                all_logprob = F.log_softmax(logits, dim=-1)  # local_rollout_forward_batch_size, seq_len, vocab_size
                logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                # log-probabilities of the generated tokens for each sequence in the batch
                del logits, all_logprob
                torch.cuda.empty_cache()

                ref_output = forward(ref_policy, query_response, tokenizer)
                ref_logits = ref_output.logits[:, context_length - 1: -1]
                ref_logits /= args.temperature + 1e-7
                ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)  # (local_rollout_forward_batch_size, gen_len)
                del ref_output, ref_logits, ref_all_logprob
                torch.cuda.empty_cache()

                # truncate response after the first occurrence of `stop_token_id` and
                # pad up to the maximum sequence length within the batch
                postprocessed_response = truncate_response(args, tokenizer, response)
                # print(f'rollout: {response=}')
                # print(f'rollout: {postprocessed_response=}')
                # run reward model on the truncated responses
                # postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                sequence_length = first_true_indices(postprocessed_response == tokenizer.pad_token_id) - 1  # (batch_size,)
                # print(f'rollout: {sequence_length=}')
                full_value, _, _ = get_reward(
                    accelerator.unwrap_model(model).critic, query_response, tokenizer, context_length
                )
                # get value estimates for generated tokens, i.e., `value`
                value = full_value[:, context_length - 1: -1].squeeze(-1)
                generated_texts = tokenizer.batch_decode(postprocessed_response,
                                                         skip_special_tokens=True)
                # print(f'rollout: {generated_texts=}')
                uam_score = compute_uam_score(generated_texts)
                score = args.sl_coef * uam_score['sl_score'] + args.wa_coef * uam_score['wa_score']
                score = score.to(device=accelerator.device)
                # _, score, _ = get_reward(reward_model, postprocessed_query_response, tokenizer, context_length)

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
            del (logprob, ref_logprob, full_value, value, score)
            torch.cuda.empty_cache()

            # Response Processing 3. filter response. Ensure that the sample contains truncate_token_id
            # responses not passing that filter will receive a low (fixed) score
            # only query humans on responses that pass that filter
            contain_eos_token = torch.any(postprocessed_responses == tokenizer.eos_token_id, dim=-1)
            if args.non_eos_penalty:
                scores = torch.where(contain_eos_token, scores, torch.full_like(scores, args.penalty_reward_value))
            accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

            # be very careful with `padding_mask_p1`
            # see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
            # values are computed for each token in the entire sequence including the previouly generated tokens
            # whereas logprobs are computed only for the generated tokens
            sequence_lengths_p1 = sequence_lengths + 1
            response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
            padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
            padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
            logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
            ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
            values = torch.masked_fill(values, padding_mask_p1, 0)

            # 4. compute rewards
            kl = logprobs - ref_logprobs  # (batch_size, gen_len)
            non_score_reward = -args.ppo.kl_coef * kl  # (batch_size, gen_len)
            rewards = non_score_reward.clone()  # (batch_size, gen_len)
            actual_start = torch.arange(rewards.size(0), device=rewards.device)  # (batch_size,)
            actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)  # (batch_size,)
            actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)  # (batch_size,)
            # add reward (from reward func) to the last position
            rewards[[actual_start, actual_end]] += scores  # (batch_size, gen_len)

            # 5. whiten rewards
            if args.ppo.whiten_rewards:
                rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
                rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

            # 6. compute advantages and returns
            lastgaelam = 0
            advantages_reversed = []
            gen_length = responses.shape[1]
            for t in reversed(range(gen_length)):
                nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                delta = rewards[:, t] + args.ppo.gamma * nextvalues - values[:, t]
                lastgaelam = delta + args.ppo.gamma * args.ppo.lam * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1], axis=1)
            returns = advantages + values
            advantages = masked_whiten(advantages, ~padding_mask)
            advantages = torch.masked_fill(advantages, padding_mask, 0)
            return_mean, return_var = returns.mean(), returns.var()
            value_mean, value_var = values.mean(), values.var()
            accelerator.print("rewards====", rewards[0])
            accelerator.print("advantages====", advantages[0])
            accelerator.print("values====", values[0])
            torch.cuda.empty_cache()

        # ppo training: iterate multiple epochs with a fresh random shuffle in each epoch
        for ppo_epoch_idx in range(args.ppo.noptepochs):
            b_inds = np.random.permutation(args.local_batch_size)
            minibatch_idx = 0
            for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                mini_batch_end = mini_batch_start + args.local_mini_batch_size
                mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                gradient_accumulation_idx = 0
                for micro_batch_start in range(0, args.local_mini_batch_size, args.local_micro_batch_size):
                    with accelerator.accumulate(policy):
                        micro_batch_end = micro_batch_start + args.local_micro_batch_size
                        micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                        mb_return = returns[micro_batch_inds]
                        mb_advantage = advantages[micro_batch_inds]
                        mb_values = values[micro_batch_inds]
                        mb_responses = responses[micro_batch_inds]
                        mb_query_responses = query_responses[micro_batch_inds]
                        mb_logprobs = logprobs[micro_batch_inds]

                        output, vpred_temp = forward(model, mb_query_responses, tokenizer)
                        logits = output.logits[:, context_length - 1 : -1]
                        logits /= args.temperature + 1e-7
                        new_all_logprobs = F.log_softmax(logits, dim=-1)
                        new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                        new_logprobs = torch.masked_fill(new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB)
                        vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                        vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds], 0)
                        vpredclipped = torch.clamp(
                            vpred,
                            mb_values - args.ppo.cliprange_value,
                            mb_values + args.ppo.cliprange_value,
                        )
                        vf_losses1 = torch.square(vpred - mb_return)
                        vf_losses2 = torch.square(vpredclipped - mb_return)
                        vf_loss_max = torch.max(vf_losses1, vf_losses2)
                        vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1[micro_batch_inds])
                        vf_clipfrac = masked_mean((vf_losses2 > vf_losses1).float(), ~padding_mask_p1[micro_batch_inds])
                        logprobs_diff = new_logprobs - mb_logprobs
                        ratio = torch.exp(logprobs_diff)
                        pg_losses = -mb_advantage * ratio
                        pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.ppo.cliprange, 1.0 + args.ppo.cliprange)
                        pg_loss_max = torch.max(pg_losses, pg_losses2)
                        pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
                        pg_clipfrac = masked_mean((pg_losses2 > pg_losses).float(), ~padding_mask[micro_batch_inds])
                        loss = pg_loss + args.ppo.vf_coef * vf_loss
                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()
                        with torch.no_grad():
                            pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
                            prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                            entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                            approxkl = 0.5 * (logprobs_diff**2).mean()
                            approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                            pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_clipfrac
                            pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                            vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                            vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_clipfrac
                            entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                            ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                    gradient_accumulation_idx += 1
                minibatch_idx += 1
                # del everything and empty cache
                # fmt: off
                del (
                    output, vpred_temp, logits, new_all_logprobs, new_logprobs, vpred, vpredclipped,
                    vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2,
                    pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,
                    mb_advantage, mb_values, mb_responses, mb_query_responses, mb_logprobs,
                )
                # fmt: on
                torch.cuda.empty_cache()
            if accelerator.is_main_process:
                console.print(
                    "ppo_epoch_idx",
                    ppo_epoch_idx,
                    "approxkl",
                    approxkl_stats[: ppo_epoch_idx + 1].mean().item(),
                    "pg_loss",
                    pg_loss_stats[: ppo_epoch_idx + 1].mean().item(),
                    "pg_clipfrac",
                    pg_clipfrac_stats[: ppo_epoch_idx + 1].mean().item(),
                    "ratio",
                    ratio_stats[: ppo_epoch_idx + 1].mean().item(),
                )
        with torch.no_grad():
            mean_kl = kl.sum(1).mean()
            mean_entropy = (-logprobs).sum(1).mean()
            mean_non_score_reward = non_score_reward.sum(1).mean()
            writer.add_scalar("objective/kl", accelerator.gather(mean_kl).mean().item(), update)
            writer.add_scalar("objective/entropy", accelerator.gather(mean_entropy).mean().item(), update)
            writer.add_scalar("objective/non_score_reward", accelerator.gather(mean_non_score_reward).mean().item(), update)
            writer.add_scalar(
                "objective/score_total", accelerator.gather(mean_non_score_reward + scores.mean()).mean().item(), update
            )
            writer.add_scalar("objective/scores", accelerator.gather(scores.mean()).mean().item(), update)
            writer.add_scalar("objective/validation_score", accelerator.gather(validation_score.mean()).mean().item(), update)
            writer.add_scalar("ppo/policy/approxkl_avg", accelerator.gather(approxkl_stats).mean().item(), update)
            writer.add_scalar("ppo/policy/clipfrac_avg", accelerator.gather(pg_clipfrac_stats).mean().item(), update)
            writer.add_scalar("ppo/loss/policy_avg", accelerator.gather(pg_loss_stats).mean().item(), update)
            writer.add_scalar("ppo/loss/value_avg", accelerator.gather(vf_loss_stats).mean().item(), update)
            writer.add_scalar("ppo/val/clipfrac_avg", accelerator.gather(vf_clipfrac_stats).mean().item(), update)
            writer.add_scalar("ppo/policy/entropy_avg", accelerator.gather(entropy_stats).mean().item(), update)
            writer.add_scalar("ppo/val/ratio", accelerator.gather(ratio_stats).mean().item(), update)
            writer.add_scalar("ppo/val/ratio_var", accelerator.gather(ratio_stats).var().item(), update)
            writer.add_scalar("ppo/val/num_eos_tokens", (responses == tokenizer.eos_token_id).sum().item(), update)
            writer.add_scalar("ppo/lr", lrnow, update)
            writer.add_scalar("ppo/episode", global_step, update)
            eps = int(global_step / (time.time() - start_time))
            writer.add_scalar("ppo/eps", eps, update)
            accelerator.print("ppo/eps", eps, update)
        del kl, mean_kl, mean_entropy, mean_non_score_reward, scores
        torch.cuda.empty_cache()

        if args.run_eval and update % args.eval_steps == 0:
            for eval_split in eval_dataloaders:
                eval_storage, eval_df = evaluate_model(
                    args.sl_coef, args.wa_coef,
                    accelerator.unwrap_model(model).policy,
                    tokenizer,
                    eval_dataloaders[eval_split],
                    validation_generation_config,
                    num_samples=args.num_eval_samples,
                )
                if args.run_eval and update % args.eval_steps == 0:
                    for eval_split in eval_dataloaders:
                        eval_storage, eval_df = evaluate_model(
                            args.sl_coef, args.wa_coef,
                            accelerator.unwrap_model(model).policy,
                            tokenizer,
                            eval_dataloaders[eval_split],
                            validation_generation_config,
                            num_samples=args.num_eval_samples,
                        )
                        if accelerator.is_main_process:
                            eval_ds = Dataset.from_pandas(eval_df)
                            # eval_ds.save_to_disk(f"runs/{args.run_name}/{eval_split}_dataset")
                            wandb.log({f"eval/{eval_split}_query_responses": wandb.Table(
                                              dataframe=eval_df)}, step=update)

                            # calculate averages
                            avg_ari = np.mean(eval_storage['ari'])
                            avg_total_score = np.mean(eval_storage['total_scores'])
                            avg_bleu = np.mean(eval_storage['bleu'])
                            avg_sent_len = np.mean(eval_storage['sent_len'])
                            avg_word_accessibility = np.mean(
                                eval_storage['word_accessibility'])
                            avg_sent_count = np.mean(eval_storage['sent_count'])

                            # log averages to wandb
                            wandb.log({
                                f"eval/{eval_split}_avg_ari": avg_ari,
                                f"eval/{eval_split}_avg_total_score": avg_total_score,
                                f"eval/{eval_split}_avg_bleu": avg_bleu,
                                f"eval/{eval_split}_avg_sent_len": avg_sent_len,
                                f"eval/{eval_split}_avg_word_accessibility": avg_word_accessibility,
                                f"eval/{eval_split}_avg_sent_count": avg_sent_count
                            }, step=update)

        # save model
        # todo: make sure there is a total limit
        if args.output_dir and args.num_train_epochs > 0 and update % args.save_steps == 0:
            avg_ari = round(np.mean(eval_storage["ari"]), 2)
            save_model(accelerator, tokenizer, model, args.output_dir,
                                avg_ari, update, args.save_total_limit)
        # if args.output_dir and args.num_train_epochs > 0 and update % args.save_steps == 0:
            # os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
            # if accelerator.is_main_process:
            #     tokenizer.save_pretrained(args.output_dir)
            #     # if args.push_to_hub:
            #     #     tokenizer.push_to_hub(repo_id=args.hf_repo_id, revision=args.hf_repo_revision)
            # unwrapped: PreTrainedModel = accelerator.unwrap_model(model).policy
            # accelerator.wait_for_everyone()
            # if accelerator.is_main_process:
            #     unwrapped.save_pretrained(
            #         args.output_dir,
            #         is_main_process=accelerator.is_main_process,
            #         save_function=accelerator.save,
            #         state_dict=accelerator.get_state_dict(unwrapped),
            #         safe_serialization=False,
            #     )
                # if args.push_to_hub:
                #     unwrapped.push_to_hub(repo_id=args.hf_repo_id, revision=args.hf_repo_revision, safe_serialization=False)
                #     accelerator.print(f"🔥 pushed to https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}")