# pylint: disable=too-few-public-methods, missing-function-docstring, no-member

"""Replay Buffer implementation"""
from typing import Iterable

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from recommender.engine.agents.panel_id_to_services_number_mapping import PANEL_ID_TO_K
from recommender.models import Sars
from recommender.engine.agents.rl_agent.preprocessing.sars_encoder import (
    SarsEncoder,
    STATE,
    USER,
    SERVICES_HISTORY,
    FILTERS,
    SEARCH_PHRASE,
    ACTION,
    REWARD,
    NEXT_STATE,
)


class ReplayBuffer(Dataset):
    """Replay Buffer used in Reinforcement Learning Actor Critic algorithm training."""

    def __init__(self, SARSes: Iterable[Sars], K: int, sars_encoder=None) -> None:
        """
        Create pytorch dataset out of SARSes.

        Args:
            SARSes: Iterable of SARSes.
            K: number of services in action (used for filtering).
            sars_encoder: SARS Encoder object.
        """

        assert K in list(PANEL_ID_TO_K.values())

        self.sars_encoder = sars_encoder
        self._load_components()

        SARSes = list(SARSes)
        SARSes = list(filter(lambda sars: len(sars.action) == K, SARSes))
        self.examples = [sars_encoder(SARS) for SARS in SARSes]

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)

    def _load_components(self):
        self.sars_encoder = self.sars_encoder or SarsEncoder()


def collate_batch(batch):
    collated_batch = {
        STATE: {
            USER: [],
            SERVICES_HISTORY: [],
            FILTERS: [],
            SEARCH_PHRASE: [],
        },
        ACTION: [],
        REWARD: [],
        NEXT_STATE: {
            USER: [],
            SERVICES_HISTORY: [],
            FILTERS: [],
            SEARCH_PHRASE: [],
        },
    }

    # Lists making
    for example in batch:
        for key1 in (STATE, NEXT_STATE):
            for key2 in (USER, SERVICES_HISTORY, FILTERS, SEARCH_PHRASE):
                collated_batch[key1][key2].append(example[key1][key2])
        collated_batch[ACTION].append(example[ACTION])
        collated_batch[REWARD].append(example[REWARD])

    # Stacking
    for key1 in (STATE, NEXT_STATE):
        for key2 in (USER, FILTERS):
            collated_batch[key1][key2] = torch.stack(collated_batch[key1][key2])
    collated_batch[ACTION] = torch.stack(collated_batch[ACTION])
    collated_batch[REWARD] = torch.stack(collated_batch[REWARD])

    # Padding
    for key1 in (STATE, NEXT_STATE):
        for key2 in (SERVICES_HISTORY, SEARCH_PHRASE):
            collated_batch[key1][key2] = pad_sequence(
                collated_batch[key1][key2], batch_first=True
            )

    return collated_batch
