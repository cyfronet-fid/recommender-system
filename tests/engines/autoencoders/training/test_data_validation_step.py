# pylint: disable-all
import pytest

from tests.engines.autoencoders.conftest import get_empty_data_and_details
from recommender.engines.autoencoders.training.data_validation_step import (
    AEDataValidationStep,
    DataValidationStep,
    DATA_IS_VALID,
)
from recommender.errors import (
    NotEnoughUsersOrServices,
    NoCategoriesScientificDomains,
)


def test_data_validation_step(
    simulate_data_extraction_step,
    delete_users_services,
    simulate_invalid_data_extraction_step,
    ae_pipeline_config,
):
    """
    Testing:
    -> configuration
    -> data validation for no users/services
    -> data validation for proper users/services
    -> data validation for invalid users/services
    """
    ae_data_validation_step = AEDataValidationStep(ae_pipeline_config)
    # Check the correctness of the configuration inside data validation step
    assert (
        ae_pipeline_config[DataValidationStep.__name__]
        == ae_data_validation_step.config
    )

    # Simulate no users/services from data extraction step
    data_ext_step, details_ext_step = get_empty_data_and_details()

    # Expect no users/services exception
    with pytest.raises(NotEnoughUsersOrServices):
        ae_data_validation_step(data_ext_step)

    # Simulate the proper data extraction step
    data_ext_step, details_ext_step = simulate_data_extraction_step

    data_valid_step, details_valid_step = ae_data_validation_step(data_ext_step)
    # Expect data to be valid
    assert details_valid_step[DATA_IS_VALID]

    # Simulate the invalid data from data extraction step
    (
        invalid_data_ext_step,
        invalid_details_ext_step,
    ) = simulate_invalid_data_extraction_step

    # Expect exception
    with pytest.raises(NoCategoriesScientificDomains):
        ae_data_validation_step(invalid_data_ext_step)
