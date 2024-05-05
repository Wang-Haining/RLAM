import csv
from datasets import load_dataset
from sacremoses import MosesTokenizer
from collections import Counter

# Load the dataset and prepare it
dataset = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)
pilot_dataset = dataset['train'].select(range(100000))  # Using a subset for pilot testing
split_dataset = pilot_dataset.train_test_split(test_size=0.05, seed=42, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')

tokenizer = MosesTokenizer(lang='en')

def process_text(text):
    """Tokenize text directly and count tokens."""
    token_counter = Counter()
    if isinstance(text, str):
        tokens = tokenizer.tokenize(text)
        token_counter.update(tokens)
    return token_counter


def process_dataset(dataset):
    """Process the dataset and accumulate token counts."""
    total_counter = Counter()
    for text in dataset['text']:
        total_counter.update(process_text(text))
    return total_counter

# Process the training and validation datasets
train_token_counter = process_dataset(split_dataset['train'])
val_token_counter = process_dataset(split_dataset['val'])

print("Training set - Total Tokens:", sum(train_token_counter.values()), "Types:", len(train_token_counter))
print("Validation set - Total Tokens:", sum(val_token_counter.values()), "Types:", len(val_token_counter))

def save_counter_to_csv(counter, filename):
    """Save a Counter object to a CSV file."""
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Token', 'Frequency'])  # Write header
        for token, frequency in counter.items():
            writer.writerow([token, frequency])

# Save the token counters to CSV files
save_counter_to_csv(train_token_counter, 'outputs/wiki_train_token_frequencies.csv')
save_counter_to_csv(val_token_counter, 'outputs/wiki_val_token_frequencies.csv')
