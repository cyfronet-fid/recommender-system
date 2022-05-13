# pylint: disable-all

from recommender.models import ML_MODELS
from recommender.services.drop_ml_models import drop_ml_models
from tests.conftest import (
    generate_users_and_services,
)
from tests.engines.ncf.conftest import mock_ncf_pipeline_exec
from tests.engines.rl.conftest import mock_rl_pipeline_exec


def test_drop_ml_models(
    generate_users_and_services, mock_ncf_pipeline_exec, mock_rl_pipeline_exec
):
    """
    Expected behaviour:
    - There are ml objects in m_l_component
    - drop_ml_models()
    - m_l_component collection is empty
    """

    for model in ML_MODELS:
        assert len(model.objects) > 0

    drop_ml_models()

    for model in ML_MODELS:
        assert model.objects.first() is None
