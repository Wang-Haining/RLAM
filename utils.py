import csv
import random
import re

import numpy as np
import pandas as pd
import syllables
import torch
from datasets import load_dataset, load_from_disk, load_metric
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from sacremoses import MosesTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer
from transformers import AutoTokenizer

PROJECT_NAME = "Reinforcement_Learning_from_Uncombined_Accessibility_Measures"
DATASET_PATH = "resources/scientific_abstract_simplification_corpus"
SEED = 42
LLAMA3_8B = 'meta-llama/Meta-Llama-3-8B'
GEMMA_2B = "google/gemma-2b"
GEMMA_7B = "google/gemma-7b"
OLMO_1B = "allenai/OLMo-1B-hf"
PHI2_3B = "microsoft/phi-2"
TASK_PREFIX = "Rewrite this abstract using simple words and short, simple sentences for middle school students: "
RESPONSE_TEMP = "\nSimplified version:"  # no space after colon because the template takes care of this
CKPTS_DIR = 'ckpts'
WORD_FREQ_CSV = "word_freq/wiki_token_freq.csv"
WORD_ACCESSIBILITY_MODEL = "word_freq/wa_model.pkl"
VOA1500 = 'word_freq/voa1500.json'
SEP_TOKENS = ['<eos>', '<|endoftext|>', '<|end_of_text|>', '<|begin_of_text|>', '<pad>']
INVALID_LOGPROB = 1.0
MAX_INPUT_LENGTHS = {'gemma-2b': 544, 'olmo-1b': 531, 'phi2-3b': 578, 'llama3-8b': 546}
MAX_OUTPUT_LENGTHS = {'gemma-2b': 241, 'olmo-1b': 223, 'phi2-3b': 244, 'llama3-8b': 240,
                      'phi-2': 244, 'OLMo-1B-hf': 223, 'Meta-Llama-3-8B': 240}


def read_token_frequencies(filename=WORD_FREQ_CSV):
    with open(filename, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # skip header
        return {rows[0]: int(rows[1]) for rows in reader}


def compute_sent_len(sent: str) -> int:
    """
    Compute length of a sentence. Punctuation marks and non-word tokens are not counted.

    Args:
        sent: A string, the input sentence to tokenize.
    Returns:
        Sentence length.
    """
    mt = MosesTokenizer(lang="en", escape=False)
    tokens = mt.tokenize(sent)
    word_pattern = re.compile(r"^'?[\w-]+$")
    return len([t for t in tokens if word_pattern.match(t)])


def compute_token_accessibility(
    token, top_100k_tokens, wa_model, total_tokens, token_freq
):
    """
    Fetch a token's accessibility score if it is among the most frequent 100,000 types
    in English Wikipedia; otherwise, estimate the accessibility using a ridge
    regression model. The accessibility score is defined as the logarithm of the token's
    frequency per billion, based on its occurrences in the English Wikipedia corpus.
    This modifies the original authors' definition of the word inaccessibility score as
    the negative logarithm.
    We adopt this approach because it is natural for a reinforcement learning model to
    maximize the gain from making a word more accessible. For example, the accessibility
    score for 'big' is 11.8, while for 'colossal' it is 7.3. Our goal is to make words
    like 'colossal' less frequent by increasing its accessibility score.

    Note,
        - We have to lowercase any token for its frequency.
        - The least frequent top_100_token is 'binion' (725 times in English wikipedia)
            or ~200/billion token.

    References:
        https://aclanthology.org/2021.ranlp-1.133/

    Args:
        token: The **lowercased** token for which the accessibility score is to be
            determined.
        top_100k_tokens: A set containing the most frequent 100,000 tokens.
        wa_model: Trained machine learning model to estimate token accessibility.
        total_tokens: Total number of tokens in the corpus for normalization.
        token_freq: A dictionary containing the occurrence of each token in the English
            Wikipedia corpus.

    Returns:
        The estimated accessibility score of the token.
    """
    if token in top_100k_tokens:
        wiki_freq = token_freq[token]
    else:
        df = pd.DataFrame({"tokens": [token.lower()], "token_len": [len(token)]})
        wiki_freq = np.exp(wa_model.predict(df)[0])
    freq_per_billion = wiki_freq / total_tokens * 1e9
    return np.log(freq_per_billion)


class ByteNGramExtractor(BaseEstimator, TransformerMixin):
    """Converts tokens into byte n-grams using a unique delimiter."""

    def __init__(self, n=1, delimiter="|"):
        self.n = n
        self.delimiter = delimiter

    def fit(self, x, y=None):
        return self

    def transform(self, tokens):
        """Transform each token into its byte n-grams, separated by a delimiter."""

        def get_byte_ngrams(token):
            bytes_token = token.encode("utf-8")
            ngrams = [
                bytes_token[i : i + self.n].decode("utf-8", "ignore")
                for i in range(len(bytes_token) - self.n + 1)
            ]
            return self.delimiter.join(ngrams)

        return [get_byte_ngrams(token) for token in tokens]


def reshape_data(x):
    """Reshape the input data to be two-dimensional."""
    return x.to_numpy().reshape(-1, 1)


def custom_analyzer(x):
    """Custom analyzer for CountVectorizer that splits on '|'."""
    return x.split("|")


def create_dataframe(data):
    """Convert training/validation data into a DataFrame."""
    return pd.DataFrame(
        {
            "tokens": [t for t, _ in data],
            "token_len": [len(t) for t, _ in data],
            "y": [np.log(freq) for _, freq in data],
        }
    )


def define_transformers():
    """Define the transformers for the column transformer."""
    return ColumnTransformer(
        [
            (
                "byte_unigrams",
                make_pipeline(
                    ByteNGramExtractor(n=1),
                    CountVectorizer(analyzer=custom_analyzer),
                ),
                "tokens",
            ),
            (
                "byte_bigrams",
                make_pipeline(
                    ByteNGramExtractor(n=2),
                    CountVectorizer(analyzer=custom_analyzer),
                ),
                "tokens",
            ),
            (
                "byte_trigrams",
                make_pipeline(
                    ByteNGramExtractor(n=3),
                    CountVectorizer(analyzer=custom_analyzer),
                ),
                "tokens",
            ),
            (
                "token_len",
                FunctionTransformer(reshape_data, validate=False),
                "token_len",
            ),
        ]
    )


def train_regression_model(train_data, val_data):
    # Prepare data
    train_df = create_dataframe(train_data)
    val_df = create_dataframe(val_data)
    # Define transformers and model pipeline
    transformer = define_transformers()
    model = Pipeline([("transformer", transformer), ("ridge", Ridge(1.0))])
    # fit the model pipeline
    X_train, y_train = train_df.drop("y", axis=1), train_df["y"]
    model.fit(X_train, y_train)
    # validate the model
    X_val = val_df.drop("y", axis=1)
    y_val = val_df["y"]
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f"MSE on validation set: {mse}")
    return model


def prepare_data(token_freq, total_tokens):
    """
    Prepare data for training or prediction by filtering out the top 50,000 most
    frequent tokens and any tokens that appear less than 50 times per billion.

    Args:
    token_freq: Dictionary with tokens as keys and their frequencies as values.
    total_tokens: Total number of tokens in the dataset.

    Returns:
        List of tuples, each containing a token, its extracted features, and its
        frequency.
    """
    min_frequency = 50 / 1e9 * total_tokens
    sorted_tokens = sorted(token_freq.items(), key=lambda item: item[1], reverse=True)
    # filter out most frequent 50k tokens
    no_top_tokens = sorted_tokens[50_000:]
    # filter out extremely rare tokens (fewer than 50 occurrences per billion)
    useful_tokens = [
        (token, freq) for token, freq in no_top_tokens if freq >= min_frequency
    ]  # 201,482 types
    return useful_tokens


def split_data(data, val_frac=0.1):
    """
    Splits data into training and validation sets.

    Args:
        data: List of tuples containing the data.
        val_frac: The fraction of data to be used for validation.

    Returns:
        tuple: Two lists, one for training and one for validation.
    """
    random.shuffle(data)
    split_idx = int(len(data) * (1 - val_frac))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    return train_data, val_data


# `is_punctuation` is adopted from
# github.com/cdimascio/py-readability-metrics/blob/master/readability/text/analyzer.py
def is_punctuation(token):
    match = re.match('^[.,\/#!$%\'\^&\*;:{}=\-_`~()]$', token)
    return match is not None


def compute_flesch_kincaid(text: str):
    """
    Compute the Flesch-Kincaid Grade Level for a given text.

    The Flesch-Kincaid Grade Level formula is: 0.39 * (total words / total sentences)
    + 11.8 * (total syllables / total words) - 15.59

    Args:
        text: The input text for which the Flesch-Kincaid Grade Level is computed.
    Returns:
        The Flesch-Kincaid Grade Level score for the input text.
    """
    # check if the last sentence is complete
    if not text.endswith((".", "?", "!")):
        # approximate the readability
        text += '.'
    mt = MosesTokenizer(lang='en', escape=False)
    sentences = sent_tokenize(text)
    words = mt.tokenize(text)
    # remove punctuation marks
    words = [w for w in words if not is_punctuation(w)]
    syllables_count = sum([syllables.estimate(word) for word in words])
    sentences_count = len(sentences)
    words_count = len(words)

    # avoid division by zero
    if sentences_count == 0 or words_count == 0:
        raise RuntimeError(f'Zero sentences/words count found in {text}')

    # apply Flesch Kincaid formula
    fk_score = (
            0.39 * (words_count / sentences_count)
            + 11.8 * (syllables_count / words_count)
            - 15.59
    )

    return fk_score


def compute_ari(text: str):
    """
    Compute the Automated Readability Index (ARI) for a given text.
    The ARI formula is: 4.71 * (characters/words) + 0.5 * (words/sentences) - 21.43
    Incomplete sentences will be concluded with an artificial period to approximate the
    ARI score.

    Args:
        text: A string of text to compute ARI.

    Returns:
        The ARI score for the input text.
    """
    # check if the last sentence is complete
    if not text.endswith((".", "?", "!")):
        # approximate the readability for incomplete sentence
        text += '.'
    mt = MosesTokenizer(lang='en', escape=False)
    sentences = sent_tokenize(text)
    words = mt.tokenize(text)
    # remove punctuation marks
    words = [w for w in words if not is_punctuation(w)]

    character_count = sum(len(word) for word in words)
    sentences_count = len(sentences)
    words_count = len(words)

    # avoid division by zero
    if sentences_count == 0 or words_count == 0:
        raise RuntimeError(f'Zero sentences/words count found in {text}.')

    # apply the ARI formula
    ari_score = (
            4.71 * (character_count / words_count)
            + 0.5 * (words_count / sentences_count)
            - 21.43
    )

    return ari_score


def count_sent(text: str) -> int:
    sents = sent_tokenize(text)
    return len(sents)


def build_ppo_dataset(
    model_name: str,
    task_prefix: str = TASK_PREFIX,
    response_template: str = RESPONSE_TEMP,
):
    """
    Build dataset for training. This function filters out too short samples and then
    extracts a specific number of samples for training.
    In the original dataset, 'source' column holds abstracts and 'target' contains
    significance statements. In the new dataset, 'query' (str) holds the templated
    abstracts.

    Args:
        model_name: base model name.
        task_prefix: The prefix to prepend to each abstract for task
        instruction.
        response_template: RESPONSE_TEMP

    Returns:
        DataLoader: The DataLoader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    ds = load_from_disk(DATASET_PATH)
    for split in ["train", "validation", "test"]:
        ds[split] = ds[split].rename_column("target", "response")
        ds[split] = ds[split].add_column("query", len(ds[split])*[''])
    def tokenize(sample):
        # prepend the task-specific prefix and append trailing template
        sample["query"] = task_prefix + sample["source"] + response_template
        if 'gemma' in model_name.lower():
            max_input_length = 544
            max_output_length = 241
        elif 'olmo' in model_name.lower():
            max_input_length = 531
            max_output_length = 223
        elif 'phi' in model_name.lower():
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
            max_input_length = 578
            max_output_length = 244
        elif 'llama' in model_name.lower():
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
            max_input_length = 546
            max_output_length = 240
        else:
            raise ValueError(f"Max lens should be computed beforehand for {model_name}.")
        query_token = tokenizer.encode(
            sample["query"],
            truncation=True,
            max_length=max_input_length,
            padding='max_length'
        )
        reference_response_token = tokenizer.encode(
            sample["response"],
            truncation=True,
            max_length=max_output_length,
            padding='max_length',
        )
        sample["query_token"] = torch.tensor(query_token)
        sample["reference_response_token"] = torch.tensor(reference_response_token)
        return sample
    ds = ds.map(tokenize, batched=False)
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def evaluate_outputs(predictions, references, sources, all_metrics=False):
    """
    Evaluate model predictions against references.

    Args:
        predictions: List of strings generated by the model.
        references: List of reference strings.
        sources: List of source strings.
        all_metrics: If True, evaluate using all metrics (ARI, BLEU, SARI, ROUGE-L),
            otherwise only ARI and BLEU.

    Returns:
        A dictionary containing metric scores.
    """
    bleu_metric = BLEU()
    results = {}

    # compute ARI
    results["ari"] = [compute_ari(p) for p in predictions]

    # compute BLEU scores
    results["bleu"] = [
        bleu_metric.corpus_bleu([p], [[r]]).score
        for p, r in zip(predictions, references)
    ]

    if all_metrics:
        # compute SARI scores
        sari_metric = load_metric("sari")
        results["sari"] = np.mean(
            [
                sari_metric.compute(predictions=[p], references=[r], sources=[s])[
                    "sari"
                ]
                for p, r, s in zip(predictions, references, sources)
            ]
        )

        # compute ROUGE-L scores
        rouge_metric = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        results["rougeL"] = [
            rouge_metric.score(p, r)["rougeL"].fmeasure
            for p, r in zip(predictions, references)
        ]

    return results
