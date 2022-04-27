# pylint: disable=missing-class-docstring

"""
Handlers used by object2vec - related functions. Used for embedding various kind of object fields.
"""

from typing import Any, List, Union, Callable

import torch
from tensorflow import Tensor
from torch.nn.utils.rnn import pad_sequence

from .text2vec import text2vec


class Handler:
    """
    Used for embedding various kind of object fields in the object2vec - related functions
    """

    def __call__(self, obj: Any) -> None:
        """
        Can be subsequently called on similar objects. It builds so called accumulator(s) that can be reduced into tensor(s) with the `get_results` function.

        Args:
            obj: Iterable of objects
        """
        ...

    def get_results(self) -> List[Tensor]:
        """
        Based on accumulator(s) build with `__call__` function it reduce it/them into list of tensors.
        """
        ...


class StringHandler(Handler):
    def __init__(self, field_name, text2vec=text2vec):
        self.field_name = field_name
        self.accumulator = []
        self.text2vec = text2vec

    def __call__(self, obj):
        self.accumulator.append(self.text2vec(getattr(obj, self.field_name)))

    def get_results(self) -> List[Tensor]:
        return [torch.stack(self.accumulator)]


class ListOfStringsHandler(Handler):
    def __init__(self, field_name, text2vec=text2vec):
        self.field_name = field_name
        self.accumulator = []
        self.text2vec = text2vec

    def __call__(self, obj):
        self.accumulator.append(self.text2vec(", ".join(getattr(obj, self.field_name))))

    def get_results(self) -> List[Tensor]:
        return [torch.stack(self.accumulator)]


class ObjectHandler(Handler):
    def __init__(self, field_name, subfields_names, text2vec=text2vec) -> None:
        self.field_name = field_name
        self.accumulators = {subfield_name: [] for subfield_name in subfields_names}
        self.text2vec = text2vec

    def __call__(self, obj) -> None:
        for subfield_name, accumulator in self.accumulators.items():
            obj = getattr(obj, self.field_name)
            accumulator.append(text2vec(getattr(obj, subfield_name)))

    def get_results(self) -> List[Tensor]:
        # There could be some flattening strategy injection
        return [torch.stack(accumulator) for accumulator in self.accumulators.values()]


class ListOfObjectsHandler(Handler):
    MAGIC_NUMBER = 512  # refactor

    def __init__(
        self,
        field_name,
        subfields_names,
        text2vec=text2vec,
        flattening_strategy: Union[Callable[[Tensor], Tensor], None] = None,
    ) -> None:
        self.field_name = field_name
        self.accumulators = {subfield_name: [] for subfield_name in subfields_names}
        self.text2vec = text2vec
        self.flattening_strategy = flattening_strategy or self._flattening_strategy

    def __call__(self, obj) -> None:
        subobjects = getattr(obj, self.field_name)
        for subfield_name, accumulator in self.accumulators.items():
            if subobjects:
                seq = []
                for subobj in subobjects:
                    subfield_val = getattr(subobj, subfield_name)
                    if subfield_val is None:
                        subfield_val = ""
                    seq.append(text2vec(subfield_val))
                tensor = torch.stack(seq)
            else:
                tensor = torch.zeros(1, self.MAGIC_NUMBER)
            accumulator.append(tensor)

    def get_results(self) -> List[Tensor]:
        return [
            self.flattening_strategy(pad_sequence(accumulator, batch_first=True))
            for accumulator in self.accumulators.values()
        ]

    def _flattening_strategy(self, padded_sequence: Tensor) -> Tensor:
        return torch.mean(padded_sequence, 1)
