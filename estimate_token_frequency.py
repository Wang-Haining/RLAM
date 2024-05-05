import pandas as pd
from datasets import load_dataset
from sacremoses import MosesTokenizer
from nltk.tokenize import sent_tokenize
from collections import Counter

dataset = load_dataset("wikipedia", "20220301.en")
# fixme
pilot_dataset = dataset['train'].select(range(100000))
split_dataset = pilot_dataset.train_test_split(test_size=0.05, seed=42, shuffle=True)
# split_dataset = dataset["train"].train_test_split(test_size=0.05, seed=42, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')

tokenizer = MosesTokenizer(lang='en')
seen_sentences = set()

# Initialize tokenizer
tokenizer = MosesTokenizer(lang='en')


def process_batch(batch):
    """Process a batch of text data to deduplicate and count tokens."""
    token_counter = Counter()
    seen_sentences = set()  # local seen sentences

    for text in batch['text']:
        if isinstance(text, str):
            sentences = sent_tokenize(text)
            for sentence in sentences:
                if sentence not in seen_sentences:
                    seen_sentences.add(sentence)
                    tokens = tokenizer.tokenize(sentence)
                    token_counter.update(tokens)
        else:
            print(f"Skipping non-string text: {text}")  # Logging non-string entries

    return token_counter


def process_dataset(dataset):
    """Process the dataset and accumulate token counts."""
    # Perform map operation to process batches
    result = dataset.map(process_batch, batched=True, batch_size=1000, num_proc=8)

    # Manually aggregate token counts from all batches
    total_counter = Counter()
    for batch_counter in result['tokens']:
        total_counter.update(batch_counter)

    return total_counter


# process training and validation datasets
train_token_counter = process_dataset(split_dataset['train'])
val_token_counter = process_dataset(split_dataset['val'])


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