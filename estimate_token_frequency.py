"""
This module estimates token frequency from the English Wikipedia for training a word
accessibility estimator to predict the frequency of an arbitrary token.
"""

import csv
import multiprocessing as mp
from collections import Counter

from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from sacremoses import MosesTokenizer
from tqdm import tqdm

# load the dataset and prepare it
dataset = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)
split_dataset = dataset["train"].train_test_split(test_size=0.05, seed=42, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')

tokenizer = MosesTokenizer(lang="en", escape=False)
result_dir = 'word_freq'
os.makedirs(result_dir, exist_ok=True)


def process_text(text):
    """Tokenize text directly and count tokens."""
    token_counter = Counter()
    sents = sent_tokenize(text)
    for sent in sents:
        tokens = tokenizer.tokenize(sent)
        token_counter.update(tokens)
    return token_counter


def process_chunk(texts):
    """Process a chunk of texts and accumulate token counts."""
    counter = Counter()
    for text in texts:
        counter.update(process_text(text.lower()))
    return counter


def process_dataset(dataset):
    """Process the dataset and accumulate token counts using multiprocessing."""
    texts = dataset["text"]
    num_processes = mp.cpu_count()
    chunksize = max(1, len(texts) // (10 * num_processes))
    # set up the pool and tqdm for the progress bar
    with mp.Pool(processes=num_processes) as pool:
        # create tasks and apply `tqdm` to the iterator for the progress bar
        tasks = [texts[i : i + chunksize] for i in range(0, len(texts), chunksize)]
        results = list(
            tqdm(
                pool.imap(process_chunk, tasks),
                total=len(tasks),
                desc="Processing Chunks",
            )
        )
        # aggregate results
        total_counter = Counter()
        for result in results:
            total_counter.update(result)
        return total_counter


# process the training and validation datasets
train_token_counter = process_dataset(split_dataset["train"])
val_token_counter = process_dataset(split_dataset["val"])

print(
    "Training set - Total Tokens:",
    sum(train_token_counter.values()),
    "Types:",
    len(train_token_counter),
)
print(
    "Validation set - Total Tokens:",
    sum(val_token_counter.values()),
    "Types:",
    len(val_token_counter),
)
# Training set - Total Tokens: 3459351600 Types: 14118606
# Validation set - Total Tokens: 181880582 Types: 2147992


def save_counter_to_csv(counter, filename):
    """Save a Counter object to a CSV file."""
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Token", "Frequency"])
        for token, frequency in counter.items():
            writer.writerow([token, frequency])


# save the token counters to CSV files
save_counter_to_csv(train_token_counter, f"{result_dir}/wiki_train_token_frequencies.csv")
save_counter_to_csv(val_token_counter, f"{result_dir}/wiki_val_token_frequencies.csv")
