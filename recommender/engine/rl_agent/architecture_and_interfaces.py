# TODO: python 3.9+ -> then typing is better: list rather than typping.List etc. (better collection support)

# General rules:
#    - zawsze gdy jakiś obiekt wymaga do inicjalizacji innych obiektów które
#      mogą być w bazie danych, to w konstruktorzy umożliwiamy bezpośrednie
#      podanie ich instancji, ale jesli nie zostaną podane to ładujemy
#      najnowszy obiekt tego typu z bazy

import random
import torch

from engine.panel_id_to_services_number_mapping import PANEL_ID_TO_K
from engine.pre_agent.pre_agent import _fill_candidate_services, _services_to_ids, InvalidRecommendationPanelIDError
from models import User, SearchData, State, Service
from typing import Dict, Tuple, Any, List

from services.fts import retrieve_services


# class BaseAgentRecommender:
#     """
#     Base Recommender class with basic functionality
#     """
#
#     def __init__(self) -> None:
#         """
#         Initialization function.
#         """
#
#         pass
#
#     def call(self, context: Dict[str, Any]) -> Tuple[int]:
#         """
#         This function allows for getting recommended services for the
#         recommendation endpoint based on recommendation context.
#
#         Args:
#             context: json dict from the /recommendations endpoint request.
#
#         Returns:
#             Tuple of recommended services ids.
#         """
#
#         self._load_models()
#         K = self._get_K(context)
#         user = self._get_user(context)
#         search_data = context.get("search_data")
#
#         if user is not None:
#             return self._for_logged_user(user, search_data, K)
#         else:
#             return self._for_anonymous_user(search_data, K)
#
#     def _load_models(self) -> None:
#         """
#         It loads model or models needed for recommendations and raise
#          exception if it is not available in the database.
#          """
#         pass
#
#     def _get_K(self, context: Dict[str, Any]) -> int:
#         """
#         Get the K constant from the context.
#
#         Args:
#             context: context json  from the /recommendations endpoint request.
#
#         Returns:
#             K constant.
#         """
#
#         K = PANEL_ID_TO_K.get(context["panel_id"])
#         if K is None:
#             raise InvalidRecommendationPanelIDError
#         return K
#
#     def _get_user(self, context: Dict[str, Any]) -> User:
#         """
#         Get the user from the context.
#
#         Args:
#             context: context json  from the /recommendations endpoint request.
#
#         Returns:
#             User.
#         """
#
#         user = None
#         if context.get("user_id"):
#             user = User.objects(id=context.get("user_id")).first()
#
#         return user
#
#     def _for_logged_user(self, user: User, search_data: SearchData, k: int) -> Tuple[int]:
#         """
#         Generate recommendation for logged user
#
#         Args:
#             user: user for whom recommendation will be generated.
#             search_data: search phrase and filters information for narrowing down an item space.
#
#         Returns:
#             Tuple of recommended services ids.
#         """
#         pass
#
#     def _for_anonymous_user(self, search_data: SearchData, K: int) -> Tuple[int]:
#         """
#         Generate recommendation for anonymous user
#
#         Args:
#             search_data: search phrase and filters information for narrowing down an item space.
#
#         Returns:
#             Tuple of recommended services ids.
#         """
#
#         candidate_services = list(retrieve_services(search_data))
#         candidate_services = _fill_candidate_services(candidate_services, K)
#         recommended_services = random.sample(list(candidate_services), K)
#         return _services_to_ids(recommended_services)


# class RLAgentRecommender(BaseAgentRecommender):
#     """
#     Reinforcement learning Agent Recommender
#     """
#
#     def __init__(self, actor_model_2=None, actor_model_3=None, state_embedder=None, action_selector=None) -> None:
#         super().__init__()
#         self.actor_model_2, self.actor_model_3 = actor_model_2, actor_model_3
#         self.state_embedder = StateEmbedder()
#         self.action_selector = ActionSelector()
#         # TODO: loading from database if None
#
#     def _load_models(self) -> None:
#         """
#         It loads model or models needed for recommendations and raise
#          exception if it is not available in the database
#         """
#
#         # TODO: implement lazy models (and maybe other things) loading
#         self.actor_model_2, self.actor_model_3 = None, None
#         pass
#
#     def _for_logged_user(self, user: User, search_data: SearchData, k: int) -> Tuple[int]:
#         """
#         Generate recommendation for logged user
#
#         Args:
#             user: user for whom recommendation will be generated.
#             search_data: search phrase and filters information for narrowing down an item space.
#
#         Returns:
#             Tuple of recommended services ids.
#         """
#
#         actor_model = self._choose_actor_model(k)
#
#         state = self._create_state(user, search_data)
#         state_tensors = self.state_embedder(state)
#
#         weights_tensor = actor_model(*state_tensors)
#         recommended_services = self.action_selector(weights_tensor, user, search_data) # TODO: jednak ids
#         recommended_services_ids = _services_to_ids(recommended_services)
#
#         return recommended_services_ids
#
#     def _choose_actor_model(self, k: int) -> torch.nn.Module:
#         """Choose appropriate model depending on the demanded recommended
#          services number"""
#
#         # TODO: implement
#         pass
#
#     def _create_state(self, user: User, search_data: SearchData) -> State:
#         """
#         Get needed information from context and create state.
#         Args:
#             context: context json  from the /recommendations endpoint request.
#
#         Returns:
#             state: State containing information about user and search_data
#         """
#
#         # TODO: implement
#         pass


# class StateEmbedder:
#     def __init__(self, user_embedder=None, service_embedder=None, search_phrase_embedder=None, filters_embedder=None):
#         self.user_embedder = user_embedder
#         self.service_embedder = service_embedder
#         self.search_phrase_embedder = search_phrase_embedder
#         self.filters_embedder = filters_embedder
#
#         # TODO: implement the rest of initialization (lazy embedders loading)
#         pass
#
#     def __call__(self, state: State) -> Tuple[torch.Tensor]:
#         """
#         Transform state to the tuple of tensors using appropriate embedders as
#          follows:
#             - state.user -> one_hot_tensor --(user_embedder)--> user_tensor
#             - state.user.accessed_services and clicks information??? -> services_one_hot_tensors --(service_embedder)--> services_dense_tensors --(concat)--> services_history_tensor
#             - state.last_search_data...filters(multiple fields) -> one_hot_tensor --(filters_embedder)--> filters_tensor
#             - state.last_search_data.q --(searchphrase_embedder)--> search_phrase_tensor
#
#         Args:
#             state: State object that contains information about user, user's services history and searchphrase.
#
#         Returns:
#             Tuple of following tensors (in this order):
#                 - user_tensor of shape [UE]
#                 - services_history_tensor of shape [N, SE]
#                 - filters_tensor of shape [FE]
#                 - search_phrase_tensor of shape [SPE]
#                 where:
#                     - UE is user content tensor embedding dim
#                     - N is user clicked services history length
#                     - SE is service content tensor embedding dim
#                     - FE is filters tensor embedding dim
#                     - SPE is search phrase tensor embedding dim
#         """
#
#         # TODO: implement
#         pass


# class HistoryEmbedder(torch.nn.Module):
#     """
#     Model used for transforming services history (list of services tensors
#      in temporal order) into history tensor. It should be used and trained inside actor and critic.
#     """
#
#     def __init__(self):
#         super().__init__()
#         self.rnn = None
#
#         # TODO: implement the rest of the initialization (layers)
#         pass
#
#     def forward(self, services_history: torch.Tensor) -> torch.Tensor:
#         """
#         RNN is used for reducing history N dimension.
#
#         Args:
#             services_history: user clicked services history tensor of shape
#              [N, SE] where N is the history length and SE is service content tensor embedding dim
#
#         Returns:
#             Clicked services history embedding tensor of shape [SE]
#         """
#
#         # TODO: implement forward computation (using RNN)
#         pass


# class ActorModel(torch.nn.Module):
#     def __init__(self, K: int, SE: int, history_embedder: torch.nn.Module):
#         super().__init__()
#         self.K = K
#         self.SE = SE
#         self.history_embedder = history_embedder
#         # WARNING: history_embedder is a model shared between actor and critic,
#         # the .detach will be probably needed for proper training
#
#         # TODO: layers initialization
#         self.output_layer = torch.nn.Linear(None, K*SE)
#
#     def forward(self,
#                 user: torch.Tensor,
#                 services_history: torch.Tensor,
#                 filters: torch.Tensor,
#                 search_phrase: torch.Tensor) -> torch.Tensor:
#         """
#         Performs forward propagation.
#
#         Args:
#             user: Embedded user content tensor of shape [UE]
#             services_history: Services history tensor of shape [N, SE]
#             filters: Embedded filters tensor of shape [FE]
#             search_phrase: Embedded search phrase tensor of shape [SPE]
#
#         """
#
#         services_history = self.history_embedder(services_history)
#
#         # TODO: implement missing forward computation
#         weights = self.output_layer(None)
#         weights = weights.reshape(self.K,self.SE)
#         return weights


# class ActionSelector:
#     def __init__(self) -> None:
#         self.itemspace = self._create_itemspace()
#
#     def __call__(self, weights: torch.Tensor, user: User, search_data: SearchData) -> Tuple[int]:
#         """
#         Based on weights_tensor, user and search_data, it selects services for
#          recommendation and create action out of them.
#
#         Args:
#             weights: Weights returned by an actor model, tensor of shape [K, SE], where:
#                 - K is the number of services required for the recommendation
#                 - SE is a service content tensor embedding dim
#             user: User for whom the recommendation is generated. User's accessed_services is used for narrowing the itemspace down.
#             search_data: Information used for narrowing the itemspace down
#
#         Returns:
#             The tuple of recommended services ids
#         """
#
#         # TODO: implement (performance is crucial)
#         pass
#
#     def _create_itemspace(self) -> torch.Tensor:
#         """
#         Creates itemspace tensor.
#
#         Returns:
#             itemspace: tensor of shape [SE, I] where: SE is service content tensor
#              embedding dim and I is the number of services
#         """
#
#         # TODO: implement
#         pass


# class ActionEmbedder:
#     def __init__(self, service_embedder=None):
#         self.service_embedder = service_embedder
#
#     def __call__(self, action: Tuple[Service]) -> Tuple[torch.Tensor]:
#         """
#         Embedd each service of the action using Service Embedder.
#
#         Args:
#             action: Tuple of services.
#
#         Returns:
#             Tuple of service embedded content tensors of shape [SE]
#         """
#
#         # TODO: implement
#         pass

# class SearchphraseEmbedder:
#     def __init__(self):
#         pass
#
#     def __call__(self, search_phrase: str) -> List[torch.Tensor]:
#         """
#         Embedd searchphrase using some embbeding
#
#         Args:
#             search_phrase: Search phrase sentence.
#
#         Returns:
#             List of sentence embedding tensors.
#         """
#
#         # TODO: implement
#         pass


# class CriticModel(torch.nn.Module):
#     def __init__(self, K: int, SE: int, history_embedder: torch.nn.Module):
#         super().__init__()
#         self.K = K
#         self.SE = SE
#         self.history_embedder = history_embedder
#         # WARNING: history_embedder is a model shared between actor and critic,
#         # the .detach will be probably needed for proper training
#
#         # TODO: layers initialization
#         self.output_layer = torch.nn.Linear(None, 1)
#
#     def forward(self, state: Tuple[torch.Tensor], action: Tuple[torch.Tensor]) -> torch.Tensor:
#         """
#         Performs forward propagation.
#
#         Args:
#             state: Tuple of following tensors:
#                 - user: Embedded user content tensor of shape [UE]
#                 - services_history: Services history tensor of shape [N, SE]
#                 - filters: Embedded filters tensor of shape [FE]
#                 - search_phrase: Embedded search phrase tensor of shape [SPE]
#                 where:
#                   - UE is user content tensor embedding dim
#                   - N is user clicked services history length
#                   - SE is service content tensor embedding dim
#                   - FE is filters tensor embedding dim
#                   - SPE is search phrase tensor embedding dim'
#
#         Returns:
#             The value of taking the given action at the given state. It's a
#              tensor of shape [] (scalar, but has be torch.Tensor to have
#              backpropagation capabilities)
#         """
#
#         user, services_history, filters, search_phrase = state
#         services_history = self.history_embedder(services_history)
#
#         # action is a tuple of K service content tensors
#         # They are not concatenated by Action Embedder because critic may want
#         # to use each of them separately in some kind of architecture, for
#         # example each tensor can be feeded into the same submodule and then
#         # results can be concatenated
#
#         # TODO: implement forward computation
#         action_value = self.output_layer(None)
#         return action_value


# class ReplayBufferGenerator:
#     # WARNING: check if services history is generated (by SARSes generator) from accessed services AND clicked services!
#     """It generates a replay buffer - dataset for the RL Agent"""
#
#     def __init__(self, state_embedder=None, action_embedder=None):
#         # TODO: implement the rest of initialization including the lazy loading
#         pass
#
#     def __call__(self) -> torch.utils.data.Dataset:
#         """
#         Generates a pytorch dataset.
#
#         Returns:
#             RL-Agent Dataset.
#
#         """
#
#         # TODO: implement
#         pass


def rl_agent_training() -> None:
    """
    This function proceed RL Agent training.

    It loads needed data from the database and perform training of the rl agent.
    In the end it save actor_model, history_embedder and critic_model to the database.

    """

    # TODO: implement training
    # TODO: models reloading callback (for pre agent also)

    pass

