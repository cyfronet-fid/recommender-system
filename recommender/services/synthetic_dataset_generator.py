# pylint: disable=no-member, missing-function-docstring, disable=missing-module-docstring

from typing import List, Tuple

import torch
import pandas as pd
from torch import nn

from recommender.models import User, Service


def embed_services(embedder: nn.Module) -> Tuple[torch.Tensor, pd.DataFrame]:
    services = Service.objects.order_by("id")
    one_hot_service_tensors = torch.Tensor([s.tensor for s in services])

    with torch.no_grad():
        embedded_services = embedder(one_hot_service_tensors)

    index_id_map = pd.DataFrame(services.distinct("id"), columns=["id"])

    return embedded_services, index_id_map


def normalize_embedded_services(embedded_services: torch.Tensor) -> torch.Tensor:
    normalization_factor = (
        2 * torch.cdist(embedded_services, torch.zeros_like(embedded_services)).max()
    )
    return embedded_services / normalization_factor


def get_service_indices(index_id_map: pd.DataFrame, ids: List[int]) -> List[int]:
    return index_id_map[index_id_map.id.isin(ids)].index.values.tolist()


def iou(set1, set2):
    return len(set1 & set2) / len(set1 | set2)


def predict_user_interest(
    user: User,
    services: List[Service],
    normalized_embedded_services: torch.Tensor,
    index_id_map: pd.DataFrame,
) -> List[float]:

    results = []

    for service in services:
        iou_categories = torch.Tensor(
            [iou(set(user.categories), set(service.categories))]
        )
        iou_scientific_domains = torch.Tensor(
            [iou(set(user.scientific_domains), set(service.scientific_domains))]
        )

        accessed_services_score = torch.mean(
            torch.Tensor([iou_categories, iou_scientific_domains])
        )

        if user.accessed_services:
            accessed_service_ids = list(map(lambda s: s.id, user.accessed_services))
            embedded_accessed_services = normalized_embedded_services[
                get_service_indices(index_id_map, accessed_service_ids)
            ]
            embedded_service = normalized_embedded_services[
                get_service_indices(index_id_map, [service.id])
            ]
            embedded_accessed_services_centroid = embedded_accessed_services.mean(dim=0)
            accessed_services_score = torch.cdist(
                embedded_service,
                embedded_accessed_services_centroid.view(1, -1),
            ).view(-1)

        results.append(
            torch.mean(
                torch.Tensor(
                    [iou_categories, iou_scientific_domains, accessed_services_score]
                )
            ).item()
        )

    return results
