# pylint: disable-all

import pytest
import random

from recommender.engines.autoencoders.training.model_validation_step import (
    AEModelValidationStep,
    ModelValidationStep,
    MODEL_IS_VALID,
    MAX_LOSS_SCORE,
)
from tests.unit.engines.autoencoders.training.test_model_evaluation_step import METRICS
from recommender.engines.autoencoders.training.model_training_step import (
    MODEL,
    EMBEDDER,
    DATASET,
)
from recommender.engines.autoencoders.training.data_extraction_step import (
    USERS,
    SERVICES,
    AUTOENCODERS,
)
from recommender.engines.autoencoders.ml_components.embedder import (
    Embedder,
    AutoEncoder,
)
from recommender.errors import PerformanceTooLowError


def test_model_validation_step(
    simulate_model_evaluation_step, ae_pipeline_config, simulate_data_preparation_step
):
    """
    Testing:
    -> configuration
    -> valid model
    -> invalid model
    """
    ae_model_valid_step = AEModelValidationStep(ae_pipeline_config)
    # Check the correctness of the configuration inside model validation step
    assert (
        ae_pipeline_config[ModelValidationStep.__name__] == ae_model_valid_step.config
    )

    data_model_eval, details_model_eval = simulate_model_evaluation_step

    # Except model_is_valid to be True
    data_model_valid, details_model_valid = ae_model_valid_step(data_model_eval)
    assert details_model_valid[MODEL_IS_VALID]

    # Test returned data
    data_prep_step, _ = simulate_data_preparation_step

    for collection in (USERS, SERVICES):
        # Check the correctness of the returned datasets on which models were trained
        assert (
            data_model_valid[collection][DATASET]
            is data_prep_step[AUTOENCODERS][collection]
        )

        # Check Embedder class memberships
        assert isinstance(data_model_valid[collection][EMBEDDER], Embedder)

        # Check AutoEncoder class memberships
        assert isinstance(data_model_valid[collection][MODEL], AutoEncoder)

    # Expect model_is_valid to be False
    config = ae_pipeline_config[ModelValidationStep.__name__]
    max_loss_score = config[MAX_LOSS_SCORE]

    random_collection = random.choice(list(data_model_eval[METRICS]))
    random_split = random.choice(list(data_model_eval[METRICS][random_collection]))
    data_model_eval[METRICS][random_collection][
        random_split
    ] = max_loss_score + random.uniform(0.001, 1)

    with pytest.raises(PerformanceTooLowError):
        data_model_valid, details_model_valid = ae_model_valid_step(data_model_eval)
        assert not details_model_valid[MODEL_IS_VALID]
