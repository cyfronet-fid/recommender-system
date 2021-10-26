# pylint: disable=fixme

"""This module contains custom Dataset class"""

from torch.utils.data.dataset import Dataset


class TensorDictDataset(Dataset):
    """Dataset wrapping named tensors as a dict."""

    def __init__(self, tensors_dict):
        first_key = next(iter(tensors_dict))
        assert all(
            tensors_dict[first_key].size(0) == tensor.size(0)
            for tensor in tensors_dict.values()
        )
        self.tensors_dict = tensors_dict

    def __getitem__(self, index):
        return {key: value[index] for key, value in self.tensors_dict.items()}

    def __len__(self):
        return list(self.tensors_dict.values())[0].size(0)
