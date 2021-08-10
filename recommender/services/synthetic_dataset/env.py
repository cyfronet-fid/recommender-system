# pylint: disable=too-many-instance-attributes, no-self-use, invalid-name

"""Module containing SyntheticMP gym ENV"""
import os
import pickle
import random
from itertools import cycle, product
from typing import List

import gym
import pandas as pd

from tqdm import tqdm

from definitions import ROOT_DIR
from recommender.models import (
    User,
    State,
    Service,
    SearchData,
    Category,
    ScientificDomain,
)
from recommender.services.fts import retrieve_services_for_recommendation

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
from recommender.services.synthetic_dataset.users import _filter_relevant


class NoSyntheticUsers(Exception):
    """Raised when the input value is too small"""


def _generate_c2xs():
    c2xs_map_path = os.path.join(ROOT_DIR, "resources", "c2xs.dict")

    if os.path.isfile(c2xs_map_path):
        with open(c2xs_map_path, "rb") as f:
            return pickle.load(f)

    categories = [None] + list(Category.objects(name__in=_filter_relevant(Category)))
    scientific_domains = [None] + list(
        ScientificDomain.objects(name__in=_filter_relevant(ScientificDomain))
    )
    c2 = list(product(categories, categories))
    c2xs = list(product(c2, scientific_domains))

    c2xs_map = {}

    for categories, domain in tqdm(c2xs):
        c = list(filter(lambda x: x is not None, categories))
        sd = list(filter(lambda x: x is not None, [domain]))

        search_data = SearchData(categories=c, scientific_domains=sd)
        services_len = len(retrieve_services_for_recommendation(search_data))

        if services_len > 2:
            key = (
                tuple(c.name if c else None for c in categories),
                domain.name if domain else None,
            )
            c2xs_map[key] = services_len

    with open("./resources/c2xs.dict", "wb") as f:
        pickle.dump(c2xs_map, f)

    return c2xs_map


def _generate_s2xc():
    s2xc_map_path = os.path.join(ROOT_DIR, "resources", "s2xc.dict")

    if os.path.isfile(s2xc_map_path):
        with open(s2xc_map_path, "rb") as f:
            return pickle.load(f)

    categories = [None] + list(Category.objects(name__in=_filter_relevant(Category)))
    scientific_domains = [None] + list(
        ScientificDomain.objects(name__in=_filter_relevant(ScientificDomain))
    )
    s2 = list(product(scientific_domains, scientific_domains))
    s2xc = list(product(s2, categories))

    s2xc_map = {}

    for domains, category in tqdm(s2xc):
        c = list(filter(lambda x: x is not None, [category]))
        sd = list(filter(lambda x: x is not None, domains))

        search_data = SearchData(scientific_domains=sd, categories=c)
        services_len = len(retrieve_services_for_recommendation(search_data))

        if services_len > 2:
            key = (
                tuple(d.name if d else None for d in domains),
                category.name if category else None,
            )
            s2xc_map[key] = services_len

    with open("./resources/s2xc.dict", "wb") as f:
        pickle.dump(s2xc_map, f)

    return s2xc_map


class SyntheticMP(gym.Env):
    """
    Simulates user interaction with EOSC Marketplace,
    assumes that the synthetic users are already generated
    """

    def __init__(
        self, interactions_per_user: int = 100, N=20, advanced_search_data=True
    ):
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

        self.advanced_search_data = advanced_search_data

        if self.advanced_search_data:
            self.c2xs_map, self.s2xc_map = _generate_c2xs(), _generate_s2xc()

    def _get_search_data(self):
        if self.advanced_search_data:
            categories = [None] + [c.name for c in self.current_user.categories]
            scientific_domains = [None] + [
                sd.name for sd in self.current_user.scientific_domains
            ]

            if random.uniform(0, 1) > 0.5:  # more categories
                c2 = list(product(categories, categories))
                c2xs = list(product(c2, scientific_domains))
                random.shuffle(c2xs)

                for subgroup in c2xs:
                    if self.c2xs_map.get(subgroup):
                        categories, domain = subgroup
                        c = list(filter(lambda x: x is not None, categories))
                        sd = list(filter(lambda x: x is not None, [domain]))
                        return SearchData(scientific_domains=sd, categories=c)

            else:  # more scientific_domains
                s2 = list(product(scientific_domains, scientific_domains))
                s2xc = list(product(s2, categories))
                random.shuffle(s2xc)

                for subgroup in s2xc:
                    if self.s2xc_map.get(subgroup):
                        domains, category = subgroup
                        c = list(filter(lambda x: x is not None, [category]))
                        sd = list(filter(lambda x: x is not None, domains))

                        return SearchData(scientific_domains=sd, categories=c)

        return SearchData()

    def _get_users(self):
        users = User.objects(synthetic=True)

        if len(users) < 1:
            raise NoSyntheticUsers

        users = list(users)
        random.shuffle(users)
        return cycle(users)

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
