# pylint: disable=too-few-public-methods, missing-function-docstring, no-member
# pylint: disable=fixme, too-many-branches

"""Replay Buffer implementation"""
import random
from collections import deque
from typing import Optional, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from recommender.engine.agents.rl_agent.preprocessing.reward_encoder import (
    RewardEncoder,
)
from recommender.engine.agents.rl_agent.preprocessing.state_encoder import StateEncoder
from recommender.engine.agents.rl_agent.preprocessing.sars_encoder import (
    STATE,
    USER,
    SERVICES_HISTORY,
    ACTION,
    REWARD,
    NEXT_STATE,
    MASK,
)

DONE = "done"


class ReplayBuffer:
    """Replay Buffer used in Reinforcement Learning Actor Critic algorithm training."""

    def __init__(
        self,
        batch_size: int,
        max_size: Union[int, float],
        state_encoder: Optional[StateEncoder] = None,
        reward_encoder: Optional[RewardEncoder] = None,
    ) -> None:
        self.state_encoder = state_encoder
        self.reward_encoder = reward_encoder
        self._load_components()

        self.batch_size = batch_size
        self._max_size = int(max_size)
        self.buffer = deque(maxlen=self._max_size)

    @property
    def max_size(self):
        return self._max_size

    @max_size.setter
    def max_size(self, new_value):
        self._max_size = int(new_value)
        self.buffer = deque(self.buffer, maxlen=self._max_size)

    def __iter__(self):
        return self

    def __next__(self):
        batch_size = min(self.batch_size, len(self.buffer))
        if batch_size == 0:
            raise Exception("Replay Buffer is Empty")

        raw_records = random.sample(self.buffer, k=batch_size)

        collated_batch = {
            STATE: {USER: [], SERVICES_HISTORY: [], MASK: []},
            ACTION: [],
            REWARD: [],
            NEXT_STATE: {USER: [], SERVICES_HISTORY: [], MASK: []},
            DONE: [],
        }

        # Lists making
        for example in raw_records:
            for key1 in (STATE, NEXT_STATE):
                for key2 in (USER, SERVICES_HISTORY, MASK):
                    collated_batch[key1][key2].append(example[key1][key2])
            for key in (ACTION, REWARD, DONE):
                collated_batch[key].append(example[key])

        # Stacking
        for key1 in (STATE, NEXT_STATE):
            for key2 in (USER, MASK):
                collated_batch[key1][key2] = torch.stack(collated_batch[key1][key2])
        for key in (ACTION, REWARD, DONE):
            collated_batch[key] = torch.stack(collated_batch[key])

        # Padding
        for key in (STATE, NEXT_STATE):
            collated_batch[key][SERVICES_HISTORY] = pad_sequence(
                collated_batch[key][SERVICES_HISTORY], batch_first=True
            )

        state = (
            collated_batch[STATE][USER],
            collated_batch[STATE][SERVICES_HISTORY],
            collated_batch[STATE][MASK],
        )
        action = collated_batch[ACTION]
        reward = collated_batch[REWARD]
        next_state = (
            collated_batch[NEXT_STATE][USER],
            collated_batch[NEXT_STATE][SERVICES_HISTORY],
            collated_batch[NEXT_STATE][MASK],
        )
        done = collated_batch[DONE]

        batch = state, action, reward, next_state, done

        return batch

    def _add_record(self, record):
        assert len(record) == 5
        state, weights, reward, next_state, done = record

        user, services_history, mask = self.state_encoder([state, next_state])
        reward = self.reward_encoder([reward])

        example = {
            STATE: {
                USER: user[0],
                SERVICES_HISTORY: services_history[0],
                MASK: mask[0],
            },
            ACTION: weights[0],
            REWARD: reward[0],
            NEXT_STATE: {
                USER: user[1],
                SERVICES_HISTORY: services_history[1],
                MASK: mask[1],
            },
            DONE: torch.tensor(float(done)),
        }

        self.buffer.append(example)

    def __lshift__(self, record):
        self._add_record(record)

    def __rshift__(self, record):
        self._add_record(record)

    def __len__(self):
        if self.buffer:
            return self.buffer[REWARD].shape[0]
        return 0

    def _load_components(self):
        self.state_encoder = self.state_encoder or StateEncoder()
        self.reward_encoder = self.reward_encoder or RewardEncoder()
