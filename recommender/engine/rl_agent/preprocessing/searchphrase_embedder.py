# pylint: disable=no-name-in-module, too-few-public-methods, fixme

"""Implementation of the Search Phrase Embedder"""

from torch import FloatTensor
from torchnlp.word_to_vector import BPEmb
from torchnlp.word_to_vector.pretrained_word_vectors import _PretrainedWordVectors


class SearchPhraseEmbedder:
    """Search phrase embedder - part of the state embedder"""

    def __init__(self, word_to_vector: _PretrainedWordVectors = None):
        """
        Args:
            word_to_vector: word_to_vector embedder, default is
             Byte-Pair Encoding (BPE) from pytorch-nlp library.
        """

        if word_to_vector is not None:
            self.word_to_vector = word_to_vector
        else:
            self.word_to_vector = BPEmb(dim=100)

    def __call__(self, search_phrase: str) -> FloatTensor:
        """
        Embed search_phrase using Byte-Pair Encoding.

        Args:
            search_phrase: Search phrase typed by user in the search box.

        Returns:
            Search phrase embedded as a tensor of shape [N, 100], where N is
             the number of search phrase's words.
        """

        words = search_phrase.split()
        embedded_search_phrase = self.word_to_vector[words]

        return embedded_search_phrase
