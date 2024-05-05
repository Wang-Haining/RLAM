import csv
from datasets import load_dataset
from sacremoses import MosesTokenizer
from nltk.tokenize import sent_tokenize
from collections import Counter

dataset = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)
# fixme
pilot_dataset = dataset['train'].select(range(100000))
split_dataset = pilot_dataset.train_test_split(test_size=0.05, seed=42, shuffle=True)
# split_dataset = dataset["train"].train_test_split(test_size=0.05, seed=42, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')

tokenizer = MosesTokenizer(lang='en')
global_seen_sentences = set()

def extract_sentences(text):
    """Extract unique sentences from a piece of text."""
    return set(sent_tokenize(text))

def tokenize_sentences(sentences):
    """Tokenize sentences and update global token counter."""
    token_counter = Counter()
    for sentence in sentences:
        if sentence not in global_seen_sentences:
            global_seen_sentences.add(sentence)
            tokens = tokenizer.tokenize(sentence)
            token_counter.update(tokens)
    return token_counter

def process_dataset(dataset):
    """Extract sentences, deduplicate globally, and count tokens."""
    all_sentences = set()
    for text in dataset['text']:
        if isinstance(text, str):
            all_sentences.update(extract_sentences(text))

    # Tokenize deduplicated sentences
    return tokenize_sentences(all_sentences)

# process training and validation datasets
train_token_counter = process_dataset(split_dataset['train'])
val_token_counter = process_dataset(split_dataset['val'])


print("Training set - Total Tokens:", sum(train_token_counter.values()), " Types:", len(train_token_counter))
print("Validation set - Total Tokens:", sum(val_token_counter.values()), "Types:", len(val_token_counter))

def save_counter_to_csv(counter, filename):
    """Save a Counter object to a CSV file."""
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Token', 'Frequency'])  # Write header
        for token, frequency in counter.items():
            writer.writerow([token, frequency])

# save DataFrames to CSV files
save_counter_to_csv(train_token_counter, 'outputs/wiki_train_token_frequencies.csv')
save_counter_to_csv(val_token_counter, 'outputs/wiki_val_token_frequencies.csv')
