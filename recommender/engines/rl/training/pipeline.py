# pylint: disable=fixme, missing-module-docstring, missing-class-docstring, invalid-name, too-few-public-methods
from recommender.engines.autoencoders.ml_components.embedder import (
    USER_EMBEDDER,
    SERVICE_EMBEDDER,
    Embedder,
)
from recommender.engines.base.base_pipeline import BasePipeline
from recommender.engines.rl.training import RLDataExtractionStep
from recommender.engines.rl.training.data_preparation_step import RLDataPreparationStep
from recommender.engines.rl.training.data_validation_step import RLDataValidationStep
from recommender.engines.rl.training.model_evaluation_step import RLModelEvaluationStep
from recommender.engines.rl.training.model_training_step import RLModelTrainingStep
from recommender.engines.rl.training.model_validation_step import RLModelValidationStep


class RLPipeline(BasePipeline):
    def _check_dependencies(self):
        Embedder.load(version=USER_EMBEDDER)
        Embedder.load(version=SERVICE_EMBEDDER)

    def _create_data_extraction_step(self) -> RLDataExtractionStep:
        return RLDataExtractionStep(self.pipeline_config)

    def _create_data_validation_step(self) -> RLDataValidationStep:
        return RLDataValidationStep(self.pipeline_config)

    def _create_data_preparation_step(self) -> RLDataPreparationStep:
        return RLDataPreparationStep(self.pipeline_config)

    def _create_model_training_step(self) -> RLModelTrainingStep:
        return RLModelTrainingStep(self.pipeline_config)

    def _create_model_evaluation_step(self) -> RLModelEvaluationStep:
        return RLModelEvaluationStep(self.pipeline_config)

    def _create_model_validation_step(self) -> RLModelValidationStep:
        return RLModelValidationStep(self.pipeline_config)
