import csv
import random
import re

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, load_from_disk, load_metric
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
from nltk.tokenize import sent_tokenize

# fixme
PROJECT_NAME = "Reinforcement Learning from Uncombined Accessibility Measures"
DATASET_PATH = "resources/scientific_abstract_simplification_corpus"
TOP_P = 0.9
SEED = 42
GEMMA = "google/gemma-2b"
OLMO = "allenai/OLMo-1B-hf"
# SEQ2SEQ_MODEL_NAME = "google/flan-t5-xl"
TASK_PREFIX = "TL;DR: "
RESPONSE_TEMP = "\nLay summary:"
WORD_FREQ_CSV = "word_freq/wiki_token_freq.csv"
WORD_ACCESSIBILITY_MODEL = "word_freq/model.pkl"


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
    mt = MosesTokenizer(lang="en")
    tokens = mt.tokenize(sent)
    word_pattern = re.compile(r"^'?[\w-]+$")
    return len([t for t in tokens if word_pattern.match(t)])


def compute_token_accessibility(
    token, top_100k_tokens, wd_model, total_tokens, token_freq
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
        wd_model: Trained machine learning model to estimate token accessibility.
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
        wiki_freq = np.exp(wd_model.predict(df)[0])
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


def compute_ari(text: str):
    """
    Compute the Automated Readability Index (ARI) for a given text.
    The ARI formula is: 4.71 * (characters/words) + 0.5 * (words/sentences) - 21.43
    Incomplete sentences will be concluded with an artificial period to approximate the
    ARI score.

    Args:
    text: A string of text to compute ARI.

    Returns:
        A list of tensors containing the processed rewards.
    """
    # check if the last sentence is complete
    if not text.endswith((".", "?", "!")):
        # approximate the readability
        text += '.'
    mt = MosesTokenizer(lang='en')
    sentences = sent_tokenize(text)
    words = mt.tokenize(text)
    # remove punctuation marks
    words = [w for w in words if not is_punctuation(w)]

    character_count = sum(len(word) for word in words)
    sentences_count = len(sentences)
    words_count = len(words)

    # avoid division by zero
    if sentences_count == 0 or words_count == 0:
        return 0

    # apply the ARI formula
    ari_score = (
            4.71 * (character_count / words_count)
            + 0.5 * (words_count / sentences_count)
            - 21.43
    )

    # clip for stability (assuming a reasonable ARI range)
    ari_score = max(min(ari_score, 35.0), 2.0)

    return ari_score
#
#
# def is_jargon(word):
#     # A lower frequency threshold means the word is less common and might be jargon
#     # fixme
#     threshold = 1e-6  # an arbitrary threshold
#     frequency = word_frequency(word, 'en')
#     return frequency < threshold


def build_dataset(
    model_name: str,
    task_prefix: str = TASK_PREFIX,
    response_template: str = RESPONSE_TEMP,
):
    """
    Build dataset for training. This function filters out too short samples and then
    extracts a specific number of samples for training.

    Args:
        model_name: SFT'ed model name.
        task_prefix: The prefix to prepend to each abstract for task
        instruction.
        response_template: RESPONSE_TEMP

    Returns:
        DataLoader: The DataLoader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = load_from_disk(DATASET_PATH)
    # fixme
    ds = ds.rename_columns({"source": "query"})

    def tokenize(sample):
        # prepend the task-specific prefix
        input_text = task_prefix + sample["query"] + response_template
        input_ids = tokenizer.encode(
            input_text,
            truncation=True,
            max_length=1024,
        )
        sample["input_ids"] = torch.tensor(input_ids)
        sample["query"] = tokenizer.decode(
            sample["input_ids"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

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

