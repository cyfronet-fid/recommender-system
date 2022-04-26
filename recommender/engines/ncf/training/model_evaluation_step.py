# pylint: disable=line-too-long

"""Neural Collaborative Filtering Model Evaluation Step."""

from typing import Tuple, Dict

import torch
from numpy import ndarray
from sklearn.metrics import classification_report
from torch.nn import BCELoss
from torch.utils.data import DataLoader

from recommender.engines.base.base_steps import ModelEvaluationStep
from recommender.engines.constants import (
    DEVICE,
    METRICS,
    LOSS,
    ACCURACY,
    CLASSIFICATION_REPORT,
)
from recommender.engines.ncf.ml_components.neural_collaborative_filtering import (
    NeuralCollaborativeFilteringModel,
)
from recommender.engines.ncf.ml_components.tensor_dict_dataset import TensorDictDataset
from recommender.engines.ncf.training.data_preparation_step import (
    DATASETS,
    TRAIN,
    VALID,
    TEST,
    USERS_IDS,
    USERS,
    SERVICES_IDS,
    SERVICES,
    LABELS,
)
from recommender.engines.ncf.training.model_training_step import (
    evaluate_ncf,
    LOSS_FUNCTION,
    BATCH_SIZE,
    MODEL,
)
from recommender.engines.metadata_creators import accuracy_function

ORDERED = "ordered"
NOT_ORDERED = "not_ordered"


def get_preds_for_ds(
    model: NeuralCollaborativeFilteringModel,
    dataset: TensorDictDataset,
    device: torch.device = torch.device("cpu"),
) -> Tuple[ndarray, ndarray]:
    """Used for getting predictions on pytorch dataset."""

    model = model.to(device)

    all_samples = dataset[:]

    users_ids = all_samples[USERS_IDS].to(device)
    users_contents = all_samples[USERS].to(device)
    services_ids = all_samples[SERVICES_IDS].to(device)
    services_contents = all_samples[SERVICES].to(device)
    labels = all_samples[LABELS].to(device)
    preds = model(users_ids, users_contents, services_ids, services_contents)

    labels = labels.detach().numpy()
    preds = preds.detach().numpy()

    return labels, preds


class NCFModelEvaluationStep(ModelEvaluationStep):
    """Neural Collaborative Filtering Model Evaluation Step."""

    def __init__(self, config):
        super().__init__(config)
        self.device = self.resolve_constant(DEVICE, torch.device("cpu"))
        self.batch_size = self.resolve_constant(BATCH_SIZE, 64)
        self.loss_function = self.resolve_constant(LOSS_FUNCTION, BCELoss())

    def __call__(self, data: Dict = None) -> Tuple[Dict, Dict]:
        """Perform evaluation of the Neural Collaborative Filtering Model."""

        model = data[MODEL]

        metrics = {
            TRAIN: {},
            VALID: {},
            TEST: {},
        }

        for ds_name, dataset in data[DATASETS].items():
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            loss, acc = evaluate_ncf(
                model=model,
                dataloader=dataloader,
                loss_function=self.loss_function,
                acc_function=accuracy_function,
                device=self.device,
            )
            metrics[ds_name][LOSS] = loss
            metrics[ds_name][ACCURACY] = acc

            y_true, y_pred = get_preds_for_ds(model, dataset)
            report = classification_report(
                y_true,
                y_pred > 0.5,
                output_dict=True,
                target_names=[ORDERED, NOT_ORDERED],
            )
            metrics[ds_name][CLASSIFICATION_REPORT] = report

        details = {METRICS: metrics}
        data[METRICS] = metrics

        return data, details
