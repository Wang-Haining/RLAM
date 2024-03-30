import re
from sacremoses import MosesTokenizer
from wordfreq import word_frequency
from nltk.tokenize import sent_tokenize



# `is_punctuation` is adopted from
# github.com/cdimascio/py-readability-metrics/blob/master/readability/text/analyzer.py
def is_punctuation(token):
    match = re.match('^[.,\/#!$%\'\^&\*;:{}=\-_`~()]$', token)
    return match is not None

def compute_ari(text: str):
    """
    Compute the Automated Readability Index (ARI) for a given text.
    The ARI formula is: 4.71 * (characters/words) + 0.5 * (words/sentences) - 21.43
    Incomplete sentences (likely not ending in a period, exclamation, or question mark)
    are not considered.

    Args:
    text: A string of text to compute ARI.

    Returns:
        A list of tensors containing the processed rewards.
    """
    mt = MosesTokenizer(lang='en')
    sentences = sent_tokenize(text)
    words = mt.tokenize(text)
    # remove punctuation marks
    words = [w for w in words if not is_punctuation(w)]

    # check if the last sentence is complete
    if sentences and not sentences[-1].endswith((".", "?", "!")):
        # remove the last sentence if it is incomplete
        sentences = sentences[:-1]

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
    threshold = 1e-6  # This threshold is arbitrary; adjust based on your needs
    frequency = word_frequency(word, 'en')
    return frequency < threshold



# mt = MosesTokenizer(lang='en')
# text = "This is an example sentence with CRISPR technology, and mRNA vaccines."
# tokens = mt.tokenize(text)
#
# # Filter and print only the tokens that are considered jargon and not punctuation
# jargon_tokens = [token for token in tokens if not is_punctuation(token) and is_jargon(token)]
# print(jargon_tokens)
