# pylint: disable=missing-function-docstring

"""This module contains Text2Vec class that encapsulate external NLP library"""

import spacy_universal_sentence_encoder
from torch import Tensor

__all__ = ["Text2Vec"]


class Text2Vec:
    """It encapsulate external NLP library"""

    def __init__(self):
        self.nlp = spacy_universal_sentence_encoder.load_model("en_use_lg")

    def __call__(self, text: str) -> Tensor:
        return Tensor(self.nlp(text).vector)

    @property
    def embedding_dim(self):
        return 512  # TODO: come on... did you think about doing it correctly?
