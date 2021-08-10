# pylint: disable=too-many-instance-attributes, no-self-use, invalid-name

"""Module containing SyntheticMP gym ENV"""
import random
from itertools import cycle
from typing import List

import gym
import pandas as pd

from recommender.models import User, State, Service, SearchData

from recommender.services.synthetic_dataset.service_engagement import (
    approx_service_engagement,
)
from recommender.services.synthetic_dataset.dataset import (
    _get_engaged_services,
    _get_ordered_services,
    _normalize_embedded_services,
)
from recommender.services.synthetic_dataset.rewards import synthesize_reward

from recommender.engine.agents.rl_agent.utils import use_service_embedder
from recommender.engine.agents.rl_agent.reward_mapping import (
    TRANSITION_REWARDS_CSV_PATH,
)
from recommender.engine.utils import load_last_module
from recommender.engine.models.autoencoders import create_embedder, SERVICES_AUTOENCODER


class NoSyntheticUsers(Exception):
    """Raised when the input value is too small"""


class SyntheticMP(gym.Env):
    """
    Simulates user interaction with EOSC Marketplace,
    assumes that the synthetic users are already generated
    """

    def __init__(self, interactions_per_user: int = 100, N=20):
        super().__init__()
        self.users = self._get_users()
        self.interactions_per_user = interactions_per_user
        self.N = N

        self.current_user = None
        self.engaged_services = None
        self.ordered_services = None
        self.interaction = None

        service_embedded_tensors, self.index_id_map = use_service_embedder(
            Service.objects.order_by("id"),
            create_embedder(load_last_module(SERVICES_AUTOENCODER)),
        )

        self.normalized_services = _normalize_embedded_services(
            service_embedded_tensors
        )
        self.transition_rewards_df = pd.read_csv(
            TRANSITION_REWARDS_CSV_PATH, index_col="source"
        )

    def _get_users(self):
        users = User.objects(synthetic=True)

        if len(users) < 1:
            raise NoSyntheticUsers

        users = list(users)
        random.shuffle(users)
        return cycle(users)

    def _get_search_data(self):
        search_categories = random.sample(
            list(self.current_user.categories),
            k=random.randint(0, len(self.current_user.categories)),
        )

        search_scientific_domains = random.sample(
            list(self.current_user.scientific_domains),
            k=random.randint(0, len(self.current_user.scientific_domains)),
        )

        return SearchData(
            categories=search_categories, scientific_domains=search_scientific_domains
        )

    def _get_state(self):
        return State(
            user=self.current_user,
            services_history=self.engaged_services[-self.N :],
            search_data=self._get_search_data(),
            synthetic=True,
        )

    def reset(self):
        self.current_user = next(self.users)
        self.ordered_services = []
        self.engaged_services = []
        self.interaction = 0

        state = self._get_state()

        return state

    def step(self, action: List[Service]):
        service_engagements = {
            s: approx_service_engagement(
                self.current_user,
                s,
                self.engaged_services,
                self.normalized_services,
                self.index_id_map,
            )
            for s in action
        }

        rewards_dict = {
            s: synthesize_reward(self.transition_rewards_df, engagement)
            for s, engagement in service_engagements.items()
        }

        self.engaged_services += _get_engaged_services(rewards_dict)
        self.ordered_services += _get_ordered_services(rewards_dict)

        state = self._get_state()
        reward = list(rewards_dict.values())
        done = self.interaction >= self.interactions_per_user

        self.interaction += 1

        return state, reward, done

    def render(self, mode="human"):
        pass
