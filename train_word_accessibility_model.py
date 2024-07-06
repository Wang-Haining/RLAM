import pickle
import random

from sacremoses import MosesTokenizer

from utils import (WORD_ACCESSIBILITY_MODEL, prepare_data, read_token_frequencies,
                   split_data, train_regression_model)

tokenizer = MosesTokenizer(lang="en", escape=False)
random.seed(42)

if __name__ == "__main__":
    # Total Tokens: 3,641,232,182 Types: 14,569,875
    token_freq = read_token_frequencies()
    total_tokens = sum(token_freq.values())

    # prepare data and train the model
    data = prepare_data(token_freq, total_tokens)
    train_data, val_data = split_data(data, val_frac=0.1)

    # val mse: Ridge ~0.444 (not sensitive to the choice of alpha)
    # OLS ~0.478, linearSVR ~0.489
    model = train_regression_model(train_data, val_data)
    pickle.dump(model, open(WORD_ACCESSIBILITY_MODEL, "wb"))
