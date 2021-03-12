# pylint: disable=too-many-locals, no-member, no-self-use, too-few-public-methods

"""This module contains Pre-Agent Dataset class"""
import random

import torch
from tqdm.auto import tqdm

from recommender.engine.pre_agent.preprocessing.common import USERS, LABELS, SERVICES
from recommender.engine.pre_agent.datasets.tensor_dict_dataset import TensorDictDataset
from recommender.models import User, Service

PRE_AGENT_DATASET = "pre-agent dataset"


class PreAgentDataset(TensorDictDataset):
    """Pre-Agent Dataset"""

    def __init__(self):
        tensors_dict = self._create_dataset()
        super().__init__(tensors_dict)

    def _create_dataset(self):
        """Creates balanced dataset that consist of pairs user-service.

        If there is n users and each of them has on average k services
        then the final dataset will consist of 2kn examples
        (not just kn because for each k positive examples of services
        oredered by a user there are generated also k negative services
        not ordered by a user).

        Time and space complexity of this algorithm is O(kn)
        """

        users_tensors = []
        services_tensors = []
        labels_tensors = []

        ordered_class_tensor = torch.Tensor([1.0])
        not_ordered_class_tensor = torch.Tensor([0.0])

        for user in tqdm(User.objects, desc="Generating dataset..."):
            # Positive examples
            ordered_services = user.accessed_services
            for service in ordered_services:
                users_tensors.append(torch.unsqueeze(torch.Tensor(user.tensor), dim=0))
                services_tensors.append(
                    torch.unsqueeze(torch.Tensor(service.tensor), dim=0)
                )
                labels_tensors.append(torch.unsqueeze(ordered_class_tensor, dim=0))

            # Negative examples (same amount as positive - classes balance)
            ordered_services_ids = [s.id for s in ordered_services]
            all_not_ordered_services = list(
                Service.objects(id__nin=ordered_services_ids)
            )
            k = min(len(ordered_services), len(all_not_ordered_services))
            not_ordered_services = random.sample(all_not_ordered_services, k=k)

            for service in not_ordered_services:
                users_tensors.append(torch.unsqueeze(torch.Tensor(user.tensor), dim=0))
                services_tensors.append(
                    torch.unsqueeze(torch.Tensor(service.tensor), dim=0)
                )
                labels_tensors.append(torch.unsqueeze(not_ordered_class_tensor, dim=0))

        users_tensor = torch.cat(users_tensors, dim=0)
        services_tensor = torch.cat(services_tensors, dim=0)
        labels_tensor = torch.cat(labels_tensors, dim=0)

        tensors_dict = {
            USERS: users_tensor,
            SERVICES: services_tensor,
            LABELS: labels_tensor,
        }

        return tensors_dict
