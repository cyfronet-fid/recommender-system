# pylint: disable=missing-function-docstring

"""Flask CLI db commands"""
from recommender.engines.rl.ml_components.sarses_generator import regenerate_sarses
from recommender.services.mp_dump import drop_mp_dump
from recommender.services.drop_ml_models import drop_ml_models
from logger_config import get_logger

logger = get_logger(__name__)


def _drop_mp():
    """Drops the documents sent by the MP Database dump"""
    drop_mp_dump()


def _drop_models():
    """Drops the documents sent by the MP Database dump"""
    drop_ml_models()


def _regenerate_sarses():
    """Regenerate SARSes"""

    logger.info("Regenerating SARSes...")
    regenerate_sarses(multi_processing=True, verbose=True)
    logger.info("SARSes regenerated successfully!")


def db_command(task):
    globals()[f"_{task}"]()
