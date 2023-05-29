"""All available engines to serve recommendations for /recommendation endpoint"""
from recommender.engines.ncf.inference.ncf_inference_component import (
    NCFInferenceComponent,
)
from recommender.engines.ncf.inference.ncf_ranking_inference_component import (
    NCFRankingInferenceComponent,
)
from recommender.engines.rl.inference.rl_inference_component import RLInferenceComponent
from recommender.engines.random.inference.random_inference_component import (
    RandomInferenceComponent,
)
from recommender.engines.random.inference.random_ranking_inference_component import (
    RandomRankingInferenceComponent,
)

# Order of engines may have an impact depending on the given context
ENGINES = {
    NCFInferenceComponent.engine_name: NCFInferenceComponent,
    RLInferenceComponent.engine_name: RLInferenceComponent,
    RandomInferenceComponent.engine_name: RandomInferenceComponent,
    NCFRankingInferenceComponent.engine_name: NCFRankingInferenceComponent,
    RandomRankingInferenceComponent.engine_name: RandomRankingInferenceComponent,
}
