# pylint: disable=no-name-in-module, too-few-public-methods, fixme

"""Implementation of the Search Phrase Embedder"""

from torch import FloatTensor
from torchnlp.word_to_vector import BPEmb
from torchnlp.word_to_vector.pretrained_word_vectors import _PretrainedWordVectors


class SearchPhraseEmbedder:
    """Search phrase embedder - part of the state embedder"""

    def __init__(self, word_to_vector: _PretrainedWordVectors = None, dim=100):
        """
        Args:
            word_to_vector: word_to_vector embedder, default is
             Byte-Pair Encoding (BPE) from pytorch-nlp library.
        """

        self.word_to_vector = word_to_vector or BPEmb(dim=dim)

    def __call__(self, search_phrase: str) -> FloatTensor:
        """
        Embed search_phrase using Byte-Pair Encoding.

        Args:
            search_phrase: Search phrase typed by user in the search box.

        Returns:
            Search phrase embedded as a tensor of shape [N, 100], where N is
             the number of search phrase's words.
        """

        # TODO: better text cleaning
        search_phrase = search_phrase.lower()

        words = search_phrase.split()
        embedded_search_phrase = self.word_to_vector[words]

        return embedded_search_phrase
