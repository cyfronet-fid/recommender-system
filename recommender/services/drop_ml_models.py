"""Function for dropping ml models from m_l_component collection"""
from recommender.models import ML_MODELS


def drop_ml_models():
    """Drops every model in m_l_component collection"""
    for model in ML_MODELS:
        for obj in model.objects:
            obj.delete()
