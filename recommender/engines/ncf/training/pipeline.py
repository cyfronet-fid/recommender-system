# pylint: disable=line-too-long, too-few-public-methods, useless-super-delegation
# pylint: disable=fixme

"""Neural Collaborative Filtering Pipeline"""

from recommender.engines.autoencoders.ml_components.embedder import (
    Embedder,
)
from recommender.engines.autoencoders.ml_components.autoencoder import (
    USER_AE_MODEL,
    SERVICE_AE_MODEL,
)
from recommender.engines.base.base_pipeline import BasePipeline
from recommender.engines.ncf.training.data_extraction_step import (
    NCFDataExtractionStep,
)
from recommender.engines.ncf.training.data_preparation_step import (
    NCFDataPreparationStep,
)
from recommender.engines.ncf.training.data_validation_step import (
    NCFDataValidationStep,
)
from recommender.engines.ncf.training.model_evaluation_step import (
    NCFModelEvaluationStep,
)
from recommender.engines.ncf.training.model_training_step import (
    NCFModelTrainingStep,
)
from recommender.engines.ncf.training.model_validation_step import (
    NCFModelValidationStep,
)


class NCFPipeline(BasePipeline):
    """Neural Collaborative Filtering Pipeline"""

    def __init__(self, pipeline_config: dict):
        super().__init__(pipeline_config)

    def _check_dependencies(self):
        Embedder.load(version=USER_AE_MODEL)
        Embedder.load(version=SERVICE_AE_MODEL)

    def _create_data_extraction_step(self) -> NCFDataExtractionStep:
        return NCFDataExtractionStep(self.pipeline_config)

    def _create_data_validation_step(self) -> NCFDataValidationStep:
        return NCFDataValidationStep(self.pipeline_config)

    def _create_data_preparation_step(self) -> NCFDataPreparationStep:
        return NCFDataPreparationStep(self.pipeline_config)

    def _create_model_training_step(self) -> NCFModelTrainingStep:
        return NCFModelTrainingStep(self.pipeline_config)

    def _create_model_evaluation_step(self) -> NCFModelEvaluationStep:
        return NCFModelEvaluationStep(self.pipeline_config)

    def _create_model_validation_step(self) -> NCFModelValidationStep:
        return NCFModelValidationStep(self.pipeline_config)
