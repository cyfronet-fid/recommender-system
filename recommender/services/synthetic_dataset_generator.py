# pylint: disable=no-member, missing-function-docstring, disable=missing-module-docstring

from typing import List

import torch
import pandas as pd

from recommender.engine.agents.rl_agent.utils import get_service_indices, iou
from recommender.models import User, Service
from recommender.services.services_history_generator import get_ordered_services


def _normalize_embedded_services(embedded_services: torch.Tensor) -> torch.Tensor:
    normalization_factor = (
        2 * torch.cdist(embedded_services, torch.zeros_like(embedded_services)).max()
    )
    output = embedded_services / normalization_factor

    return output


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

        ordered_services = get_ordered_services(user)

        if ordered_services:
            ordered_service_ids = list(map(lambda s: s.id, ordered_services))
            embedded_ordered_services = normalized_embedded_services[
                get_service_indices(index_id_map, ordered_service_ids)
            ]
            embedded_service = normalized_embedded_services[
                get_service_indices(index_id_map, [service.id])
            ]
            embedded_ordered_services_centroid = embedded_ordered_services.mean(dim=0)
            accessed_services_score = torch.cdist(
                embedded_service,
                embedded_ordered_services_centroid.view(1, -1),
            ).view(-1)

        results.append(
            torch.mean(
                torch.Tensor(
                    [iou_categories, iou_scientific_domains, accessed_services_score]
                )
            ).item()
        )

    return results
