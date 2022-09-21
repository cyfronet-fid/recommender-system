# pylint: disable-all

from recommender.engines.engines import ENGINES
from recommender.engines.explanations import Explanation


def test_all_engine_classes_have_default_explanation():
    assert all(
        isinstance(engine_class.default_explanation, Explanation)
        for engine_class in ENGINES.values()
    )
