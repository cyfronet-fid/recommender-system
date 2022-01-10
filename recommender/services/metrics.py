# pylint: disable-all

"""High level recommendations metrics"""

from typing import Tuple
from tqdm.auto import tqdm
from recommender.models import UserAction
from recommender.models import Recommendation


def get_clicks(recommendation: Recommendation) -> Tuple[int, int]:
    """Get panel/not panel clicks number from the given recommendation"""

    panel_clicks = UserAction.objects(
        source__visit_id=recommendation.visit_id,
        source__root__type__="recommendation_panel",
    ).count()

    not_panel_clicks = UserAction.objects(
        source__visit_id=recommendation.visit_id,
        source__root__type__ne="recommendation_panel",
    ).count()

    return panel_clicks, not_panel_clicks


def calc_hitrate_for(recommendations, verbose=True):
    """Calculate grouped hitrate factors for the given recommendations
    sequence"""

    panel_clicks_acc, not_panel_clicks_acc = 0, 0
    if recommendations.count() == 0:
        return 0, 0
    prog_bar = tqdm(recommendations, disable=not verbose)
    for recommendation in prog_bar:
        panel_clicks, not_panel = get_clicks(recommendation)
        panel_clicks_acc += panel_clicks
        not_panel_clicks_acc += not_panel
        hitrate = panel_clicks_acc / (panel_clicks_acc + not_panel_clicks_acc) * 100
        prog_bar.set_postfix(
            {
                "panel": panel_clicks_acc,
                "not panel": not_panel_clicks_acc,
                "hitrate": f"{hitrate:.4f}%",
            }
        )

    return panel_clicks_acc, not_panel_clicks_acc, hitrate


def calc_hitrate(engine_version, panel_id):
    """Calculate grouped hitrate factors for recommendations that match given
    parameters"""
    panel_clicks, not_panel_clicks, hitrate = calc_hitrate_for(
        Recommendation.objects(engine_version=engine_version, panel_id=panel_id),
        verbose=True,
    )
    print(
        f"RESULTS [panel:{panel_clicks},"
        f" not panel:{not_panel_clicks},"
        f" hitrate:{hitrate:.4f}]\n"
    )
