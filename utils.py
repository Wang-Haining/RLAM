import re
import torch
import numpy as np
from datasets import load_dataset, load_from_disk
from rouge_score import rouge_scorer
from datasets import load_metric
from sacrebleu.metrics import BLEU
from sacremoses import MosesTokenizer
from wordfreq import word_frequency
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
from joblib import load
from sklearn.base import BaseEstimator, TransformerMixin

# fixme
BASELINE_MODEL = "haining/sas_baseline"
DATASET_PATH = 'resources/scientific_abstract_simplification_corpus'
TOP_P = 0.9
SEED = 42
PROJECT_NAME = 'Scholarly_Abstract_Simplification'
CLM_MODEL_NAME = "google/gemma-2b"
SEQ2SEQ_MODEL_NAME = 'google/flan-t5-xl'
TASK_PREFIX = ("Simplify the scholarly abstract so it is immediately understandable "
               "to a layperson: ")
RESPONSE_TEMP = "\nA concise lay summary:"

T5_MAX_INPUT_LEN = 512
T5_MAX_OUTPUT_LEN = 256

# word_difficulty_model = load('word_freq/model.joblib')
def compute_sent_len(sent: str) -> int:
    """
    Compute length of a sentence. Punctuation marks and non-word tokens are not counted.

    Args:
        sent: A string, the input sentence to tokenize.
    Returns:
        Sentence length.
    """
    mt = MosesTokenizer(lang='en')
    tokens = mt.tokenize(sent)
    word_pattern = re.compile(r"^'?[\w-]+$")
    return len([t for t in tokens if word_pattern.match(t)])



# def predict_token_difficulty(token, model):
#     """Predict the difficulty of a token using a pre-trained model."""
#     df = pd.DataFrame({
#         'tokens': [token],
#         'token_len': [len(token)]
#     })
#
#     return model.predict(df).pop()

    # # Example prediction
    # token = 'example'
    # difficulty = predict_token_difficulty(token, model)
    # print(f"Difficulty for token '{token}': {difficulty}")


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


def compute_token_difficulty():
    """

    Word difficulty is defined as the negative log frequency in the English Wikipedia
    corpus.

    :return:
    """


    type = []
    pass



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


def is_jargon(word):
    # A lower frequency threshold means the word is less common and might be jargon
    # fixme
    threshold = 1e-6  # an arbitrary threshold
    frequency = word_frequency(word, 'en')
    return frequency < threshold


def build_dataset(
        model_name: str,
        task_prefix: str = TASK_PREFIX,
        response_template: str = RESPONSE_TEMP
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
        sample["query"] = tokenizer.decode(sample["input_ids"],
                                           skip_special_tokens=True,
                                           clean_up_tokenization_spaces=True)

        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def evaluate_outputs(predictions,
                     references,
                     sources,
                     all_metrics=False):
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
    results['ari'] = [compute_ari(p) for p in predictions]

    # compute BLEU scores
    results['bleu'] = [bleu_metric.corpus_bleu([p], [[r]]).score for p, r in
                       zip(predictions, references)]

    if all_metrics:
        # compute SARI scores
        sari_metric = load_metric('sari')
        results['sari'] = np.mean([sari_metric.compute(predictions=[p],
                                                       references=[r],
                                                       sources=[s])['sari'] for p, r, s
                                   in zip(predictions, references, sources)])

        # compute ROUGE-L scores
        rouge_metric = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        results['rougeL'] = [rouge_metric.score(p, r)['rougeL'].fmeasure for p, r in
                             zip(predictions, references)]

    return results

# mt = MosesTokenizer(lang='en')
# text = "This is an example sentence with CRISPR technology, and mRNA vaccines."
# tokens = mt.tokenize(text)
#
# # Filter and print only the tokens that are considered jargon and not punctuation
# jargon_tokens = [token for token in tokens if not is_punctuation(token) and
# is_jargon(token)]
# print(jargon_tokens)
