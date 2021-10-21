# pylint: disable-all

# TODO: Fixup synthetic env
# """Module containing SyntheticMP gym ENV"""
# import os
# import pickle
# import random
# from copy import deepcopy
# from itertools import cycle, product
# from typing import List
#
# import gym
# import pandas as pd
# from torch.utils.tensorboard import SummaryWriter
#
# from tqdm import tqdm
#
# from definitions import ROOT_DIR, LOG_DIR
# from recommender.models import (
#     User,
#     State,
#     Service,
#     SearchData,
#     Category,
#     ScientificDomain,
# )
# from recommender.services.fts import retrieve_services_for_recommendation
#
# from recommender.services.synthetic_dataset.service_engagement import (
#     approx_service_engagement,
# )
# from recommender.services.synthetic_dataset.dataset import (
#     _get_engaged_services,
#     _get_ordered_services,
#     _normalize_embedded_services,
# )
# from recommender.services.synthetic_dataset.rewards import synthesize_reward
#
# from recommender.engine.agents.rl_agent.utils import use_service_embedder
# from recommender.engine.agents.rl_agent.reward_mapping import (
#     TRANSITION_REWARDS_CSV_PATH,
# )
# from recommender.engine.utils import load_last_module
# from recommender.engine.models.autoencoders import create_embedder, SERVICES_AUTOENCODER
# from recommender.services.synthetic_dataset.users import _filter_relevant
#
#
# class NoSyntheticUsers(Exception):
#     """Raised when the input value is too small"""
#
#
# def _generate_c2xs():
#     c2xs_map_path = os.path.join(ROOT_DIR, "resources", "c2xs.dict")
#
#     if os.path.isfile(c2xs_map_path):
#         with open(c2xs_map_path, "rb") as f:
#             return pickle.load(f)
#
#     categories = [None] + list(Category.objects(name__in=_filter_relevant(Category)))
#     scientific_domains = [None] + list(
#         ScientificDomain.objects(name__in=_filter_relevant(ScientificDomain))
#     )
#     c2 = list(product(categories, categories))
#     c2xs = list(product(c2, scientific_domains))
#
#     c2xs_map = {}
#
#     for categories, domain in tqdm(c2xs):
#         c = list(filter(lambda x: x is not None, categories))
#         sd = list(filter(lambda x: x is not None, [domain]))
#
#         search_data = SearchData(categories=c, scientific_domains=sd)
#         services_len = len(retrieve_services_for_recommendation(search_data))
#
#         if services_len > 2:
#             key = (
#                 tuple(c.id if c else -1 for c in categories),
#                 domain.id if domain else -1,
#             )
#             c2xs_map[key] = services_len
#
#     with open("./resources/c2xs.dict", "wb") as f:
#         pickle.dump(c2xs_map, f)
#
#     return c2xs_map
#
#
# def _generate_s2xc():
#     s2xc_map_path = os.path.join(ROOT_DIR, "resources", "s2xc.dict")
#
#     if os.path.isfile(s2xc_map_path):
#         with open(s2xc_map_path, "rb") as f:
#             return pickle.load(f)
#
#     categories = [None] + list(Category.objects(name__in=_filter_relevant(Category)))
#     scientific_domains = [None] + list(
#         ScientificDomain.objects(name__in=_filter_relevant(ScientificDomain))
#     )
#     s2 = list(product(scientific_domains, scientific_domains))
#     s2xc = list(product(s2, categories))
#
#     s2xc_map = {}
#
#     for domains, category in tqdm(s2xc):
#         c = list(filter(lambda x: x is not None, [category]))
#         sd = list(filter(lambda x: x is not None, domains))
#
#         search_data = SearchData(scientific_domains=sd, categories=c)
#         services_len = len(retrieve_services_for_recommendation(search_data))
#
#         if services_len > 2:
#             key = (
#                 tuple(d.id if d else -1 for d in domains),
#                 category.id if category else -1,
#             )
#             s2xc_map[key] = services_len
#
#     with open("./resources/s2xc.dict", "wb") as f:
#         pickle.dump(s2xc_map, f)
#
#     return s2xc_map
#
#
# class SyntheticMP(gym.Env):
#     """
#     Simulates user interaction with EOSC Marketplace,
#     assumes that the synthetic users are already generated
#     """
#
#     def __init__(
#         self,
#         interactions_per_user: int = 100,
#         N=20,
#         advanced_search_data=True,
#         max_depth=10,
#         users_n=1,
#     ):
#         super().__init__()
#         self.max_depth = max_depth
#         self.interactions_per_user = interactions_per_user
#         self.N = N
#
#         self.indices = None
#         self.current_user_idx = None
#         self.current_user = None
#         self.engaged_services = None
#         self.current_engaged_services = None
#         self.ordered_services = None
#         self.interaction = None
#
#         self.users_n = users_n
#
#         self.users = self._get_users()
#
#         service_embedded_tensors, self.index_id_map = use_service_embedder(
#             Service.objects.order_by("id"),
#             create_embedder(load_last_module(SERVICES_AUTOENCODER)),
#         )
#
#         self.normalized_services = _normalize_embedded_services(
#             service_embedded_tensors
#         )
#         self.transition_rewards_df = pd.read_csv(
#             TRANSITION_REWARDS_CSV_PATH, index_col="source"
#         )
#
#         self.advanced_search_data = advanced_search_data
#
#         if self.advanced_search_data:
#             self.c2xs_map, self.s2xc_map = _generate_c2xs(), _generate_s2xc()
#
#         self.counter = 0
#
#     def _get_search_data(self):
#         if self.advanced_search_data:
#             categories = [-1] + [c.id for c in self.current_user.categories]
#             scientific_domains = [-1] + [
#                 sd.id for sd in self.current_user.scientific_domains
#             ]
#
#             if random.uniform(0, 1) > 0.5:  # more categories
#                 c2 = list(product(categories, categories))
#                 c2xs = list(product(c2, scientific_domains))
#                 random.shuffle(c2xs)
#
#                 for subgroup in c2xs:
#                     if self.c2xs_map.get(subgroup):
#                         category_ids, domain_id = subgroup
#                         categories = Category.objects(id__in=category_ids)
#                         if domain_id != -1:
#                             chosen_scientific_domain_ids = [domain_id] + random.sample(
#                                 scientific_domains,
#                                 k=random.randint(0, len(scientific_domains)),
#                             )
#                             scientific_domains = ScientificDomain.objects(
#                                 id__in=chosen_scientific_domain_ids
#                             )
#                         else:
#                             scientific_domains = []
#
#                         return SearchData(
#                             scientific_domains=scientific_domains, categories=categories
#                         )
#
#             else:  # more scientific_domains
#                 s2 = list(product(scientific_domains, scientific_domains))
#                 s2xc = list(product(s2, categories))
#                 random.shuffle(s2xc)
#
#                 for subgroup in s2xc:
#                     if self.s2xc_map.get(subgroup):
#                         domain_ids, category_id = subgroup
#                         scientific_domains = ScientificDomain.objects(id__in=domain_ids)
#                         if category_id == -1:
#                             chosen_category_ids = [category_id] + random.sample(
#                                 categories, k=random.randint(0, len(categories))
#                             )
#                             categories = ScientificDomain.objects(
#                                 id__in=chosen_category_ids
#                             )
#                         else:
#                             categories = []
#
#                         return SearchData(
#                             scientific_domains=scientific_domains, categories=categories
#                         )
#
#         return SearchData()
#
#     def _get_users(self):
#         users = User.objects(synthetic=True)[: self.users_n]
#
#         if len(users) < 1:
#             raise NoSyntheticUsers
#
#         users = list(users)
#
#         # TODO: refactor
#         indices = list(range(len(users)))
#         random.shuffle(indices)
#         self.indices = deepcopy(cycle(indices))
#         self.current_user_idx = next(self.indices)
#         self.engaged_services = [[] for _ in range(len(users))]
#
#         return users
#
#     def _get_state(self):
#         return State(
#             user=self.current_user,
#             services_history=self.current_engaged_services[-self.N :],
#             search_data=self._get_search_data(),
#             synthetic=True,
#         )
#
#     def reset(self):
#         self.current_user_idx = next(self.indices)
#         self.current_user = self.users[self.current_user_idx]
#         self.current_engaged_services = self.engaged_services[self.current_user_idx]
#         self.ordered_services = []
#         self.interaction = 0
#
#         state = self._get_state()
#
#         return state
#
#     def step(self, action: List[Service]):
#         if action is None:
#             return None, None, None, {}
#
#         service_engagements = {
#             s: approx_service_engagement(
#                 self.current_user,
#                 s,
#                 self.current_engaged_services,
#                 self.normalized_services,
#                 self.index_id_map,
#             )
#             for s in action
#         }
#
#         d = {
#             f"{i}": engagement
#             for i, engagement in enumerate(service_engagements.values())
#         }
#
#         rewards_dict = {
#             s: synthesize_reward(
#                 self.transition_rewards_df, engagement, max_depth=self.max_depth
#             )
#             for s, engagement in service_engagements.items()
#         }
#
#         # TODO: below filtering may potentially break something in the future
#         new_engaged_services = _get_engaged_services(rewards_dict)
#         # self.current_engaged_services += new_engaged_services
#         self.current_engaged_services += list(
#             set(new_engaged_services) - set(self.current_engaged_services)
#         )
#         self.engaged_services[self.current_user_idx] = self.current_engaged_services
#
#         # self.ordered_services += _get_ordered_services(rewards_dict) # TODO: uncomment
#         state = self._get_state()
#         reward = list(rewards_dict.values())
#         done = self.interaction >= self.interactions_per_user
#
#         self.interaction += 1
#
#         self.counter += 1
#
#         return state, reward, done, {}
#
#     def render(self, mode="human"):
#         pass
