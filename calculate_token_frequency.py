"""
This module is used to calculate the frequency of tokens (defined by a Moses tokenizer)
in the English Wikipedia corpus.
"""

import os
import csv
import multiprocessing as mp
from collections import Counter

from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from sacremoses import MosesTokenizer
from tqdm import tqdm

from utils import WORD_FREQ_CSV

dataset = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)
tokenizer = MosesTokenizer(lang="en")


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


def save_counter_to_csv(counter, filename):
    """Save a Counter object to a CSV file, ensuring the parent directory exists."""
    # ensure the parent directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Token", "Frequency"])
        for token, frequency in counter.items():
            writer.writerow([token, frequency])


token_counter = process_dataset(dataset["train"])


print(
    "Total Tokens:",
    sum(token_counter.values()),
    "Types:",
    len(token_counter),
)

save_counter_to_csv(token_counter, WORD_FREQ_CSV)
