import csv
import random

import numpy as np
import pandas as pd
from joblib import dump
from sacremoses import MosesTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer

tokenizer = MosesTokenizer(lang="en")


def read_token_frequencies(filename):
    with open(filename, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # skip header
        return {rows[0]: int(rows[1]) for rows in reader}


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


def train_regression_model(train_data, val_data):
    train_df = pd.DataFrame(
        {
            "tokens": [t for t, _ in train_data],
            "token_len": [len(t) for t, _ in train_data],
            "y": [np.log(freq) for _, freq in train_data],
        }
    )
    val_df = pd.DataFrame(
        {
            "tokens": [t for t, _ in val_data],
            "token_len": [len(t) for t, _ in val_data],
            "y": [np.log(freq) for _, freq in val_data],
        }
    )

    # fixme: transformer.named_transformers_['byte_unigrams'].named_steps['countvectorizer'].get_feature_names_out()
    transformer = ColumnTransformer(
        [
            (
                "byte_unigrams",
                make_pipeline(
                    ByteNGramExtractor(n=1),
                    CountVectorizer(analyzer=lambda x: x.split("|")),
                ),
                "tokens",
            ),
            (
                "byte_bigrams",
                make_pipeline(
                    ByteNGramExtractor(n=2),
                    CountVectorizer(analyzer=lambda x: x.split("|")),
                ),
                "tokens",
            ),
            (
                "byte_trigrams",
                make_pipeline(
                    ByteNGramExtractor(n=3),
                    CountVectorizer(analyzer=lambda x: x.split("|")),
                ),
                "tokens",
            ),
            (
                "token_len",
                FunctionTransformer(
                    lambda x: x.to_numpy().reshape(-1, 1), validate=False
                ),
                "token_len",
            ),
        ]
    )

    model = Pipeline([("transformer", transformer), ("ridge", Ridge())])

    # fit the model pipeline
    X = train_df.drop("y", axis=1)
    y = train_df["y"]
    model.fit(X, y)

    y_pred = model.predict(val_df.drop("y", axis=1))
    mse = mean_squared_error(val_df["y"], y_pred)
    print(f"MSE on validation set: {mse}")
    dump(model, "word_freq/model.joblib")
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


# def predict_token_difficulty(token, model):
#     """Predict the difficulty of a token using a pre-trained model."""
#     df = pd.DataFrame({
#         'tokens': [token],
#         'token_len': [len(token)]
#     })
#
#     return model.predict(df).pop()


if __name__ == "__main__":
    # Total Tokens: 3641232182 Types: 14569875
    token_freq = read_token_frequencies("word_freq/wiki_token_freq.csv")
    total_tokens = sum(token_freq.values())

    # prepare data and train the model
    data = prepare_data(token_freq, total_tokens)
    train_data, val_data = split_data(data, val_frac=0.1)
    model = train_regression_model(data)

    # # Example prediction
    # token = 'example'
    # difficulty = predict_token_difficulty(token, model)
    # print(f"Difficulty for token '{token}': {difficulty}")
