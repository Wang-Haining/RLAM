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
print(f'{config=}')

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

print(dataset)
print(dataset.column_names)
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name)
ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)



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
    "min_length": 20,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 512,
}


# max_ppo_steps = 500

for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    # if step >= max_ppo_steps:
    #     break
    query_tensors = batch["input_ids"]
    response_tensors = []
    print(f'{len(query_tensors)=}')
    for query in query_tensors:  # batch_size (1, 512)
        gen_len = output_length_sampler()
        # generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query.squeeze(0), **generation_kwargs) # 1, max_new_tokens
        response_tensors.append(response.squeeze()[-gen_len:])  # batch_size (max_new_tokens,)
    batch["response"] = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]  # #batch_size of str
    print(f'{batch["response"]=}')

    # Normalize the rewards and ensure the reward tensors are on the correct device
    raw_rewards = [compute_ari(r) * (-1.0) for r in batch["response"]]

    mean_reward = np.mean(raw_rewards)
    std_reward = np.std(raw_rewards)
    normalized_rewards = [(r - mean_reward) / (std_reward + 1e-9) for r in raw_rewards]
    reward_tensors = [torch.tensor(r, dtype=torch.float32) for r in normalized_rewards]

    # Execute a PPO step
    # query_tensors (mini_batch_size * batch_size), ?
    # response_tensors (mini_batch_size * batch_size), gen_len
    stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
    ppo_trainer.log_stats(stats, batch, reward_tensors)

