# pylint: disable-all

import pytest

from recommender.engines.autoencoders.training.model_evaluation_step import (
    AEModelEvaluationStep,
    ModelEvaluationStep,
    METRICS,
)
from recommender.engines.autoencoders.training.data_preparation_step import (
    VALID,
    TRAIN,
    TEST,
)
from recommender.engines.autoencoders.training.data_extraction_step import (
    USERS,
    SERVICES,
    AUTOENCODERS,
)
from recommender.engines.autoencoders.training.model_training_step import (
    MODEL,
    EMBEDDER,
    DATASET,
)
from recommender.engines.autoencoders.ml_components.embedder import (
    Embedder,
    AutoEncoder,
)


def test_model_evaluation_step(
    simulate_model_training_step,
    training_data,
    ae_pipeline_config,
    simulate_data_preparation_step,
):
    """
    Testing:
    -> configuration
    -> metrics in data
    -> metrics in details
    """
    ae_model_eval_step = AEModelEvaluationStep(ae_pipeline_config)
    # Check the correctness of the configuration inside model evaluation step
    assert ae_pipeline_config[ModelEvaluationStep.__name__] == ae_model_eval_step.config

    data_model_train_step, _ = simulate_model_training_step
    data_model_eval_step, details_model_eval_step = ae_model_eval_step(
        data_model_train_step
    )

    data_prep_step, _ = simulate_data_preparation_step
    # Data
    for collection in (USERS, SERVICES):
        # Check the correctness of the returned datasets on which models were trained
        assert (
            data_model_eval_step[collection][DATASET]
            is data_prep_step[AUTOENCODERS][collection]
        )

        # Check Embedder class memberships
        assert isinstance(data_model_eval_step[collection][EMBEDDER], Embedder)

        # Check AutoEncoder class memberships
        assert isinstance(data_model_eval_step[collection][MODEL], AutoEncoder)

    data_model_eval_step = data_model_eval_step[METRICS]
    details_model_eval_step = details_model_eval_step[METRICS]

    # Data and details
    for collection in (USERS, SERVICES):
        for split in (TRAIN, VALID, TEST):
            # Check if metrics from the same collection and split are the same in data and details
            assert (
                data_model_eval_step[collection][split]
                is details_model_eval_step[collection][split]
            )

            for metric in (data_model_eval_step, details_model_eval_step):
                loss = metric[collection][split]
                assert isinstance(loss, float)
