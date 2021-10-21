# pylint: disable=line-too-long, too-few-public-methods, useless-super-delegation
"""Autoencoder pipeline"""

from recommender.engines.base.base_pipeline import BasePipeline
from recommender.engines.autoencoders.training.data_extraction_step import (
    AEDataExtractionStep,
)
from recommender.engines.autoencoders.training.data_preparation_step import (
    AEDataPreparationStep,
)
from recommender.engines.autoencoders.training.data_validation_step import (
    AEDataValidationStep,
)
from recommender.engines.autoencoders.training.model_evaluation_step import (
    AEModelEvaluationStep,
)
from recommender.engines.autoencoders.training.model_training_step import (
    AEModelTrainingStep,
)
from recommender.engines.autoencoders.training.model_validation_step import (
    AEModelValidationStep,
)
from recommender.models.pipeline_metadata import PipelineMetadata


class AEPipeline(BasePipeline):
    """Autoencoders Pipeline"""

    def __init__(self, pipeline_config: dict):
        super().__init__(pipeline_config)

    def _create_metadata(self) -> PipelineMetadata:
        return PipelineMetadata(
            type=f"{self.__class__.__name__}",
        )

    def _check_dependencies(self):
        pass

    def _create_data_extraction_step(self) -> AEDataExtractionStep:
        return AEDataExtractionStep(self.pipeline_config)

    def _create_data_validation_step(self) -> AEDataValidationStep:
        return AEDataValidationStep(self.pipeline_config)

    def _create_data_preparation_step(self) -> AEDataPreparationStep:
        return AEDataPreparationStep(self.pipeline_config)

    def _create_model_training_step(self) -> AEModelTrainingStep:
        return AEModelTrainingStep(self.pipeline_config)

    def _create_model_evaluation_step(self) -> AEModelEvaluationStep:
        return AEModelEvaluationStep(self.pipeline_config)

    def _create_model_validation_step(self) -> AEModelValidationStep:
        return AEModelValidationStep(self.pipeline_config)
