import re
import pandas as pd
from datasets import load_dataset
from sacremoses import MosesTokenizer
from nltk.tokenize import sent_tokenize
from collections import Counter

dataset = load_dataset("wikipedia", "20220301.en")

split_dataset = dataset["train"].train_test_split(test_size=0.05, seed=42, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')

tokenizer = MosesTokenizer(lang='en')
seen_sentences = set()


def process_batch(batch, seen_sentences):
    """Process a batch of text data to deduplicate and count tokens."""
    token_counter = Counter()

    for text in batch['text']:
        sentences = sent_tokenize(text)
        for sentence in sentences:
            if sentence not in seen_sentences:
                seen_sentences.add(sentence)
                tokens = tokenizer.tokenize(sentence)
                token_counter.update(tokens)

    return token_counter


def process_dataset(dataset, seen_sentences):
    """Process the entire dataset in batches, using a global deduplication set."""
    token_counter = Counter()
    for batch in dataset.batch(1000):  # process in batches of 1000 examples
        batch_counter = process_batch(batch, seen_sentences)
        token_counter.update(batch_counter)
    return token_counter


# process training and validation datasets
train_token_counter = process_dataset(split_dataset['train'], seen_sentences)
val_token_counter = process_dataset(split_dataset['val'], seen_sentences)

print("Training set - Total Tokens:", sum(train_token_counter.values()), " Types:", len(train_token_counter))
print("Validation set - Total Tokens:", sum(val_token_counter.values()), "Types:", len(val_token_counter))

train_df = pd.DataFrame.from_dict(train_token_counter, orient='index', columns=['Frequency'])
val_df = pd.DataFrame.from_dict(val_token_counter, orient='index', columns=['Frequency'])

# save DataFrames to CSV files
train_df.to_csv('outputs/wiki_train_token_frequencies.csv')
val_df.to_csv('outputs/wiki_val_token_frequencies.csv')


# def compute_word_freq():
#     """
#
#     Word difficulty is defined as the negative log frequency in the English Wikipedia
#     corpus.
#
#     :return:
#     """
#
#
#     type = []
#     pass