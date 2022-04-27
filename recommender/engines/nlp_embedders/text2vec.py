import spacy_universal_sentence_encoder
from torch import Tensor

__all__ = ["text2vec"]

_nlp = spacy_universal_sentence_encoder.load_model("en_use_lg")


def text2vec(text: str) -> Tensor:
    return Tensor(_nlp(text).vector)
