import csv
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
from sacremoses import MosesTokenizer
from collections import Counter
import multiprocessing as mp
from tqdm import tqdm

# Load the dataset and prepare it
dataset = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)
pilot_dataset = dataset['train'].select(
    range(100000))  # Using a subset for pilot testing
split_dataset = pilot_dataset.train_test_split(test_size=0.05, seed=42, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')

tokenizer = MosesTokenizer(lang='en')


def process_text(text):
    """Tokenize text directly and count tokens."""
    token_counter = Counter()
    if isinstance(text, str):
        sents = sent_tokenize(text)
        for sent in sents:
            tokens = tokenizer.tokenize(sent)
            token_counter.update(tokens)
    return token_counter


def process_chunk(texts):
    """Process a chunk of texts and accumulate token counts."""
    counter = Counter()
    for text in texts:
        counter.update(process_text(text))
    return counter


def process_dataset(dataset):
    """Process the dataset and accumulate token counts using multiprocessing."""
    texts = [t.lower() for t in dataset['text']]
    num_processes = mp.cpu_count()
    chunksize = max(1, len(texts) // (10 * num_processes))
    # set up the pool and tqdm for the progress bar
    with mp.Pool(processes=num_processes) as pool:
        # create tasks and apply `tqdm` to the iterator for the progress bar
        tasks = [texts[i:i + chunksize] for i in range(0, len(texts), chunksize)]
        results = list(tqdm(pool.imap(process_chunk, tasks), total=len(tasks),
                            desc="Processing Chunks"))
        # aggregate results
        total_counter = Counter()
        for result in results:
            total_counter.update(result)
        return total_counter


# process the training and validation datasets
train_token_counter = process_dataset(split_dataset['train'])
val_token_counter = process_dataset(split_dataset['val'])

print("Training set - Total Tokens:", sum(train_token_counter.values()), "Types:",
      len(train_token_counter))
print("Validation set - Total Tokens:", sum(val_token_counter.values()), "Types:",
      len(val_token_counter))


def save_counter_to_csv(counter, filename):
    """Save a Counter object to a CSV file."""
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Token', 'Frequency'])
        for token, frequency in counter.items():
            writer.writerow([token, frequency])


# Save the token counters to CSV files
save_counter_to_csv(train_token_counter, 'outputs/wiki_train_token_frequencies.csv')
save_counter_to_csv(val_token_counter, 'outputs/wiki_val_token_frequencies.csv')
