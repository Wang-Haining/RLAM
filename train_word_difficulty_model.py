import csv
import os.path
import pickle
import random

import numpy as np
import pandas as pd
from sacremoses import MosesTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer
from utils import WORD_DIFFICULTY_MODEL

tokenizer = MosesTokenizer(lang="en")
random.seed(42)

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


def read_token_frequencies(filename):
    with open(filename, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # skip header
        return {rows[0]: int(rows[1]) for rows in reader}


def reshape_data(x):
    """Reshape the input data to be two-dimensional."""
    return x.to_numpy().reshape(-1, 1)


def custom_analyzer(x):
    """Custom analyzer for CountVectorizer that splits on '|'."""
    return x.split('|')


def create_dataframe(data):
    """Convert training/validation data into a DataFrame."""
    return pd.DataFrame({
        "tokens": [t for t, _ in data],
        "token_len": [len(t) for t, _ in data],
        "y": [np.log(freq) for _, freq in data]
    })


def define_transformers():
    """Define the transformers for the column transformer."""
    return ColumnTransformer([
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
    ])


def train_regression_model(train_data, val_data):
    # Prepare data
    train_df = create_dataframe(train_data)
    val_df = create_dataframe(val_data)
    # Define transformers and model pipeline
    transformer = define_transformers()
    model = Pipeline([
        ("transformer", transformer),
        ("ridge", Ridge(1.0))
    ])
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


# if __name__ == "__main__":
if not os.path.exists(WORD_DIFFICULTY_MODEL):
    # Total Tokens: 3,641,232,182 Types: 14,569,875
    token_freq = read_token_frequencies(WORD_DIFFICULTY_MODEL)
    total_tokens = sum(token_freq.values())

    # prepare data and train the model
    data = prepare_data(token_freq, total_tokens)
    train_data, val_data = split_data(data, val_frac=0.1)
    # val mse: Ridge 0.444 (not sensitive to the choice of alpha)
    # OLS 0.478, linearSVR 0.489
    model = train_regression_model(train_data, val_data)
    # save the trained model
    with open(WORD_DIFFICULTY_MODEL, 'wb') as file:
        pickle.dump(model, file)
else:
    model = pickle.load(WORD_DIFFICULTY_MODEL)