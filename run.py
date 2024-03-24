import torch
from tqdm import tqdm
import numpy as np

from nltk.tokenize import word_tokenize, sent_tokenize

from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from trl.core import LengthSampler
import wandb


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Set the device to GPU 1
device = torch.device("cuda:0")

wandb.init()

learning_rate = 1.41e-5
max_ppo_epochs = 1
mini_batch_size = 2
batch_size = 8
# model_name = "haining/sas_baseline"
model_name = "google/flan-t5-small"

config = PPOConfig(
    model_name=model_name,
    learning_rate=learning_rate,
    ppo_epochs=max_ppo_epochs,
    mini_batch_size=mini_batch_size,
    batch_size=batch_size,
    log_with="wandb",
)
#
# config = PPOConfig(
#     # model_name="lvwerra/gpt2-imdb",
#     model_name="haining/sas_baseline",
#     # model_name="google-t5/t5-small",
#     learning_rate=1.41e-5,
#     log_with="wandb",
#     ppo_epochs=1,
#     mini_batch_size=4,
#     batch_size=8,
# )


def build_dataset(config,
                  dataset_name="gfissore/arxiv-abstracts-2021",
                  task_prefix="summarize, simplify, and contextualize: ",
                  num_samples=10000):
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
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    ds = load_dataset(dataset_name, split="train")

    # Assuming the column with abstracts is named 'abstract'
    ds = ds.rename_columns({"abstract": "text"})

    # Filter out too short samples first
    ds = ds.filter(lambda x: len(x["text"]) > 20, batched=False)

    # Then select the first num_samples
    ds = ds.select(range(min(num_samples, len(ds))))

    def tokenize(sample):
        #                 prompt = f"""
        # summarize, simplify, and contextualize: {sample["dialogue"]}
        # Summary:
        # """
        # Prepend the task-specific prefix
        input_text = task_prefix + sample["text"]
        # Tokenize, ensuring the result is within the model's context window
        input_ids = tokenizer(input_text, return_tensors='pt', truncation=True,
                              padding='max_length',
                              max_length=tokenizer.model_max_length).input_ids
        sample["input_ids"] = input_ids
        sample["query"] = tokenizer.decode(input_ids.squeeze())
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    # Split the dataset into train and test parts.
    ds_splits = ds.train_test_split(test_size=0.05, shuffle=False, seed=42)

    # return dataset_splits
    return ds_splits


dataset = build_dataset(config)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name)
ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name)
ppo_model.to(device)
ref_model.to(device)

tokenizer = AutoTokenizer.from_pretrained(config.model_name)

# tokenizer.pad_token = tokenizer.eos_token

ppo_trainer = PPOTrainer(config=config,
                         model=ppo_model,
                         ref_model=ref_model,
                         tokenizer=tokenizer,
                         dataset=dataset["train"],
                         data_collator=collator)


def compute_ari(text):
    """
    Compute the Automated Readability Index (ARI) for a given text.
    The ARI formula is: 4.71 * (characters/words) + 0.5 * (words/sentences) - 21.43
    Incomplete sentences (likely not ending in a period, exclamation, or question mark) are not considered.
    """
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    # Check if the last sentence is complete
    if sentences and not sentences[-1].endswith(('.', '?', '!')):
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
    ari_score = 4.71 * (characters / words_count) + 0.5 * (
                words_count / sentences_count) - 21.43
    return ari_score


# todo: investigate
output_min_length = 20
output_max_length = 512
output_length_sampler = LengthSampler(output_min_length, output_max_length)

generation_kwargs = {
    "min_length": 5,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": -1,
    # "batch_size": 1
}


max_ppo_steps = 500

for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    if step >= max_ppo_steps:
        break

    query_tensors = batch["input_ids"]
    response_tensors = []
    raw_rewards = []

    for query in query_tensors:
        print(f"Query tensor shape before generate: {query.shape=}")
        max_new_tokens = output_length_sampler()
        generation_kwargs["max_new_tokens"] = max_new_tokens

        # Ensure the query tensor is on the correct device before generation
        query = query.squeeze(0).to(device)
        # query = query.to(device)

        print(f"Query tensor shape after transformation: {query.shape=}")
        response = ppo_trainer.generate(query, **generation_kwargs)
        print(f"{response.shape=}")
        # # Compute the reward using ARI
        decoded_response = tokenizer.decode(response.squeeze(), skip_special_tokens=True)
        # print(f"{type(decoded_response)=}")
        # print(f"{decoded_response=}")
        ari_reward = compute_ari(decoded_response) * (-1.0)
        raw_rewards.append(ari_reward)

        response_tensors.append(response.squeeze()[-max_new_tokens:])

    # Update response in batch with the proper decoding and on the correct device
    batch["response"] = [tokenizer.decode(r.to(device).squeeze()) for r in response_tensors]

    # Normalize the rewards and ensure the reward tensors are on the correct device
    mean_reward = np.mean(raw_rewards)
    std_reward = np.std(raw_rewards)
    normalized_rewards = [(r - mean_reward) / (std_reward + 1e-9) for r in raw_rewards]
    reward_tensors = [torch.tensor([reward], dtype=torch.float32, device=device) for
                      reward in normalized_rewards]

    # Execute a PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
    ppo_trainer.log_stats(stats, batch, reward_tensors)

