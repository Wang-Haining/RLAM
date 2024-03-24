import os

import numpy as np
import torch
import wandb
from datasets import load_dataset
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
from trl import AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

wandb.init()

learning_rate = 1.41e-5
max_ppo_epochs = 1
mini_batch_size = 4
batch_size = 16
model_name = "haining/sas_baseline"

config = PPOConfig(
    model_name=model_name,
    learning_rate=learning_rate,
    ppo_epochs=max_ppo_epochs,
    mini_batch_size=mini_batch_size,
    batch_size=batch_size,
    log_with="wandb",
)


def build_dataset(
    config,
    dataset_name="gfissore/arxiv-abstracts-2021",
    task_prefix="summarize, simplify, and contextualize: ",
    num_samples=10000,
):
    """
    Build dataset for training with FLAN-T5. This function filters out too short samples
    and then extracts a specific number of samples for training.

    Args:
        config: Configuration object containing the model name and other parameters.
        dataset_name (str, optional): The name of the dataset to be loaded.
        task_prefix (str, optional): The prefix to prepend to each abstract for task
        instruction.
        num_samples (int, optional): The number of samples to be extracted from the
        dataset for training.

    Returns:
        DataLoader: The DataLoader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"abstract": "query"})
    ds = ds.filter(lambda x: len(x["query"]) > 20, batched=False)

    # Then select the first num_samples
    ds = ds.select(range(min(num_samples, len(ds))))

    def tokenize(sample):
        # # Prepend the task-specific prefix
        input_text = task_prefix + sample["query"]
        # Tokenize without returning tensors
        input_ids = tokenizer.encode(
            input_text,
            truncation=True,
            # padding=False,
            # max_length=tokenizer.model_max_length,
        )
        # Convert list of input_ids to a 1D tensor
        sample["input_ids"] = torch.tensor(input_ids)

        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    # Split the dataset into train and test parts.
    ds_splits = ds.train_test_split(test_size=0.05, shuffle=False, seed=42)

    # return dataset_splits
    return ds_splits


def normalize_rewards_and_convert_to_tensors(responses, compute_ari_func):
    """
    Normalize rewards calculated from the responses using the Automated Readability Index (ARI)
    and convert them into tensors.

    Args:
        responses (list of str): The responses to compute the rewards for.
        compute_ari_func (callable): Function used to compute the ARI score for a response.

    Returns:
        list of torch.Tensor: The list of normalized reward tensors.
    """
    # Calculate raw rewards using ARI function and invert them (since we're assuming
    # that lower ARI is better, hence * -1.0)
    raw_rewards = [compute_ari_func(r) * (-1.0) for r in responses]

    # Normalize the raw rewards
    mean_reward = np.mean(raw_rewards)
    std_reward = np.std(raw_rewards)
    normalized_rewards = [(r - mean_reward) / (std_reward + 1e-9) for r in raw_rewards]

    # Convert the normalized rewards to tensors
    reward_tensors = [torch.tensor(r, dtype=torch.float32) for r in normalized_rewards]

    return reward_tensors


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def compute_ari(text):
    """
    Compute the Automated Readability Index (ARI) for a given text.
    The ARI formula is: 4.71 * (characters/words) + 0.5 * (words/sentences) - 21.43
    Incomplete sentences (likely not ending in a period, exclamation, or question mark)
    are not considered.
    """
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    # Check if the last sentence is complete
    if sentences and not sentences[-1].endswith((".", "?", "!")):
        # Remove the last sentence if it is incomplete
        sentences = sentences[:-1]

    characters = sum(len(word) for word in words)

    # Calculate the number of sentences and words
    sentences_count = len(sentences)
    words_count = len(words)

    # Avoid division by zero
    if sentences_count == 0 or words_count == 0:
        return 0

    # Apply the ARI formula
    ari_score = (
        4.71 * (characters / words_count)
        + 0.5 * (words_count / sentences_count)
        - 21.43
    )
    return ari_score


if __name__ == "__main__":

    ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name)
    ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    dataset = build_dataset(config)
    ppo_trainer = PPOTrainer(
        config=config,
        model=ppo_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset["train"],
        data_collator=collator,
    )

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": 512,
    }

    for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        response_tensors, ref_response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            generate_ref_response=True,
            **generation_kwargs
        )
        batch["response"] = tokenizer.batch_decode(response_tensors)
        batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

        # normalize the rewards and ensure the reward tensors are on the correct device
        reward = normalize_rewards_and_convert_to_tensors(
            batch["response"], compute_ari
        )
        # ref rewards
        ref_reward = normalize_rewards_and_convert_to_tensors(
            batch["ref_response"], compute_ari
        )
        batch["ref_rewards"] = ref_reward

        # execute a PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, reward)
        ppo_trainer.log_stats(
            stats,
            batch,
            reward,
            columns_to_log=["query", "response", "ref_response", "ref_rewards"],
        )
