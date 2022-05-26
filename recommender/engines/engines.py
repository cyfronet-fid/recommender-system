"""All available engines to serve recommendations for /recommendation endpoint"""
from recommender.engines.ncf.inference.ncf_inference_component import (
    NCFInferenceComponent,
)
from recommender.engines.rl.inference.rl_inference_component import RLInferenceComponent
from recommender.engines.random.inference.random_inference_component import (
    RandomInferenceComponent,
)

# Order of engines may have an impact depending on the given context
ENGINES = {
    RLInferenceComponent.engine_name: RLInferenceComponent,
    NCFInferenceComponent.engine_name: NCFInferenceComponent,
    RandomInferenceComponent.engine_name: RandomInferenceComponent,
}
