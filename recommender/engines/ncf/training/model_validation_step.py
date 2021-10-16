# pylint: disable=line-too-long

"""Neural Collaborative Filtering Model Validation step"""

import time
from typing import Tuple, Dict

import torch
from torch import Tensor

from recommender.engines.base.base_steps import ModelValidationStep
from recommender.engines.ncf.ml_components.neural_collaborative_filtering import (
    NeuralCollaborativeFilteringModel,
)
from recommender.engines.ncf.training.data_preparation_step import (
    USERS_IDS,
    USERS,
    SERVICES_IDS,
    SERVICES,
    DATASETS,
    TRAIN,
    TEST,
)
from recommender.engines.ncf.ml_components.tensor_dict_dataset import (
    TensorDictDataset,
)
from recommender.engines.ncf.training.model_evaluation_step import (
    METRICS,
    CLASSIFICATION_REPORT,
)
from recommender.engines.ncf.training.model_training_step import (
    MODEL,
)
from recommender.errors import InferenceTooSlowError, PerformanceTooLowError

MAX_EXECUTION_TIME = "max_execution_time"
MAX_ITEMSPACE_SIZE = "max_itemspace_size"
MIN_WEIGHTED_AVG_F1_SCORE = "min_weighted_avg_f1_score"
MODEL_IS_VALID = "model_is_valid"


def prepare_testing_sample(
    data: Dict[str, Dict[str, TensorDictDataset]],
    max_itemspace_size: int,
    device: torch.device,
) -> Dict[str, Tensor]:
    """Prepare testing sample out of provided data.

    Args:
        data: Any ncf dataset,
        max_itemspace_size: Representative size of the serving itemspace,
        device: device for tensors casting.

    Returns:
        testing_sample: Batch of data ready for inference.
    """
    dataset = data[DATASETS][TRAIN]
    all_samples = dataset[:]

    users_contents = all_samples[USERS][0].repeat(max_itemspace_size, 1).to(device)
    users_ids = (
        all_samples[USERS_IDS][0].repeat(max_itemspace_size, 1).reshape(-1).to(device)
    )
    services_contents = (
        all_samples[SERVICES][0].repeat(max_itemspace_size, 1).to(device)
    )
    services_ids = (
        all_samples[SERVICES_IDS][0]
        .repeat(max_itemspace_size, 1)
        .reshape(-1)
        .to(device)
    )

    testing_sample = {
        "users_contents": users_contents,
        USERS_IDS: users_ids,
        "services_contents": services_contents,
        SERVICES_IDS: services_ids,
    }

    return testing_sample


def measure_execution_time(
    model: NeuralCollaborativeFilteringModel, testing_sample: Dict[str, Tensor]
) -> float:
    """Measure time of the model inference on testing sample.

    Args:
        model: NCF model,
        testing_sample: Batch of data ready for inference.

    Returns:
        execution_time: Batch of data ready for inference.
    """

    start = time.time()
    model(**testing_sample)
    end = time.time()

    execution_time = end - start

    return execution_time


def check_speed(
    data: Dict[str, Dict[str, TensorDictDataset]],
    max_execution_time: float,
    max_itemspace_size: int,
) -> None:
    """Check the speed of the model.

    Args:
        data: data from the NCFModelEvaluationStep,
        max_execution_time: threshold for time,
        max_itemspace_size: itemspace size for the testing sample preparation.

    """

    device = torch.device("cpu")  # Because serving is done on CPU
    model = data[MODEL].to(device)

    testing_sample = prepare_testing_sample(data, max_itemspace_size, device)
    execution_time = measure_execution_time(model, testing_sample)

    if execution_time >= max_execution_time:
        raise InferenceTooSlowError()


def check_performance(
    data: Dict[str, Dict[str, TensorDictDataset]], min_weighted_avg_f1_score: float
) -> None:
    """Check if the F1-score of the model is below certain threshold.

    Args:
        data: data from the NCFModelEvaluationStep,
        min_weighted_avg_f1_score: threshold used for validation.

    """
    weighted_avg_f1_score = data[METRICS][TEST][CLASSIFICATION_REPORT]["weighted avg"][
        "f1-score"
    ]
    if weighted_avg_f1_score < min_weighted_avg_f1_score:
        raise PerformanceTooLowError()


class NCFModelValidationStep(ModelValidationStep):
    """Neural Collaborative Filtering Model Validation step"""

    def __init__(self, config):
        super().__init__(config)
        self.max_execution_time = self.resolve_constant(MAX_EXECUTION_TIME, 0.1)
        self.max_itemspace_size = self.resolve_constant(MAX_ITEMSPACE_SIZE, 10000)
        self.min_weighted_avg_f1_score = self.resolve_constant(
            MIN_WEIGHTED_AVG_F1_SCORE, 0.6
        )

    def __call__(self, data: Dict = None) -> Tuple[Dict, Dict]:
        """Perform model validation consisting of checking:
        -> model speed,
        -> model performance regarding chosen metric.
        """
        details = {MODEL_IS_VALID: False}

        check_speed(data, self.max_execution_time, self.max_itemspace_size)
        check_performance(data, self.min_weighted_avg_f1_score)

        details[MODEL_IS_VALID] = True

        return data, details
