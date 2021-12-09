# pylint: disable-all

"""User Journey module"""

import random
import uuid
from copy import deepcopy

from mongoengine import connect, disconnect
from pymongo import uri_parser

from tests.factories.marketplace import UserFactory
from tests.factories.recommendation import RecommendationFactory
from tests.factories.user_action import UserActionFactory
from recommender.models import Recommendation, UserAction
from settings import TestingConfig


class LowLevelUserJourney:
    # TODO: Refactor whole class

    def __init__(self, anonymous=False, user=None, unique_id=None, panel_id="v1"):
        if anonymous:
            if user is not None:
                raise Exception(
                    "Provided user for anonymous user. For anonymous user only"
                    " unique_id can be provided."
                )
            unique_id = unique_id or uuid.uuid4()
            self.user_params = {"user": None, "unique_id": unique_id}
        else:
            if unique_id is not None:
                raise Exception(
                    "Provided unique_id for logged user. For logged user only"
                    " user can be provided."
                )
            user = user or UserFactory()
            self.user_params = {"user": user, "unique_id": None}
        self.panel_id = {panel_id: True}

        ua = UserActionFactory(**self.user_params)
        rec = RecommendationFactory(
            **self.user_params, **self.panel_id, visit_id=ua.target.visit_id
        )
        self.last_visit_id = rec.visit_id

    def go_to_rec_page(self, start_ua=None):
        """ua-->recommendation"""

        if start_ua is None:
            start_visit_id = self.last_visit_id
        else:
            start_visit_id = start_ua.target.visit_id

        ua = UserActionFactory(**self.user_params, source__visit_id=start_visit_id)
        recommendation = RecommendationFactory(
            **self.user_params, **self.panel_id, visit_id=ua.target.visit_id
        )
        self.last_visit_id = recommendation.visit_id

        return recommendation

    def uas_chain(self, n: int, start_ua: UserAction = None) -> UserAction:
        """ua-->ua-->ua-->...-->ua"""

        assert n > 0

        if start_ua is None:
            visit_id = self.last_visit_id
        else:
            visit_id = start_ua.target.visit_id

        ua = None
        for _ in range(n):
            ua = UserActionFactory(**self.user_params, source__visit_id=visit_id)
            visit_id = ua.target.visit_id

        self.last_visit_id = ua.target.visit_id

        return ua

    def click_random_service_from_rec(self, recommendation):
        k = random.randint(0, len(recommendation.services) - 1)
        return self.click_service_from_rec(k, recommendation)

    def click_service_from_rec(self, k, recommendation):
        """
                             +-->service1-->[ua]
                           /
            recommendation+----->service2-->[ua]
                          \
                           +---->service3-->[ua]
        """
        try:
            service = recommendation.services[k]
        except IndexError as e:
            raise Exception(
                f"This recommendation has {len(recommendation.services)}"
                f" services. Choose appropriate index."
            ) from e

        ua = UserActionFactory(
            **self.user_params,
            source__visit_id=recommendation.visit_id,
            recommendation_root=True,
            source__root__service=service,
        )
        self.last_visit_id = ua.target.visit_id

        return ua

    def click_random_ua_from_rec_tree(self, recommendation, order=False, n=1):
        """
                             +-->ua
                           /
            recommendation+---->ua-->ua-->ua-->ua
                          \            \
                           +-->ua       +-->ua
        """

        all_uas = self._get_all_uas_of_the_rec(recommendation)
        if len(all_uas) > 0:
            for _ in range(n):
                ua = random.choice(list(all_uas))
                ua = UserActionFactory(
                    **self.user_params, order=order, source__visit_id=ua.target.visit_id
                )
        else:
            if order:
                raise Exception("Can't order directly from recommendation panel")
            for _ in range(n):
                ua = UserActionFactory(
                    **self.user_params,
                    order=order,
                    source__visit_id=recommendation.visit_id,
                )
        self.last_visit_id = ua.target.visit_id

        return ua

    def _get_all_uas_of_the_rec(self, rec):
        """get all uas of the rec, but do not traverse through order actions"""

        all_uas = set()

        last_uas = set(
            UserAction.objects(**self.user_params, source__visit_id=rec.visit_id)
        )
        all_uas |= last_uas
        while last_uas:
            next_uas = set()
            for rec_ua in last_uas:
                next_uas |= set(
                    UserAction.objects(
                        **self.user_params, source__visit_id=rec_ua.target.visit_id
                    )
                )
            all_uas |= next_uas
            last_uas = set(
                filter(
                    lambda ua: not ua.action.order
                    and Recommendation.objects(
                        **self.user_params, visit_id=ua.target.visit_id
                    ).first()
                    is not None,
                    next_uas,
                )
            )

        return all_uas

    def order(self, start_ua=None):
        """ua-->order_ua"""

        if start_ua is None:
            rec = Recommendation.objects(visit_id=self.last_visit_id).first()
            if rec is not None:
                ua = UserActionFactory(
                    **self.user_params,
                    source__visit_id=rec.visit_id,
                    recommendation_root=True,
                    source__root__service=random.choice(rec.services),
                )
                ua = UserActionFactory(
                    **self.user_params, order=True, source__visit_id=ua.target.visit_id
                )

            else:
                ua = UserActionFactory(
                    **self.user_params, order=True, source__visit_id=self.last_visit_id
                )
        else:
            ua = UserActionFactory(
                **self.user_params,
                order=True,
                source__visit_id=start_ua.target.visit_id,
            )
        self.last_visit_id = ua.target.visit_id

        return ua

    def _get_last_rec(self):
        return Recommendation.objects(**self.user_params).order_by("-timestamp").first()

    def _get_last_user_action(self):
        return UserAction.objects(**self.user_params).order_by("-timestamp").first()


def the_testing_db_context(func):
    def func_with_testing_db_context(*args, **kwargs):
        uri = TestingConfig.MONGODB_HOST
        db = connect(host=uri)

        func(*args, **kwargs)

        db_name = uri_parser.parse_uri(uri)["database"]
        db.drop_database(db_name)
        disconnect()

    return func_with_testing_db_context


class UserJourney:
    def __init__(self, anonymous=False, user=None, unique_id=None, panel_id="v1"):
        self.lluj = LowLevelUserJourney(
            anonymous=anonymous, user=user, unique_id=unique_id, panel_id=panel_id
        )

    def next(self, n: int = 1) -> "UserJourney":
        new_stage = deepcopy(self)
        new_stage.lluj.uas_chain(n=n)
        return new_stage

    def order(self):
        new_stage = deepcopy(self)
        new_stage.lluj.order()
        return new_stage

    def go_to_panel(self):
        new_stage = deepcopy(self)
        new_stage.lluj.go_to_rec_page()
        return new_stage

    def service(self, k: int = None):
        new_stage = deepcopy(self)
        recommendation = Recommendation.objects(
            visit_id=new_stage.lluj.last_visit_id
        ).first()
        if recommendation is None:
            raise Exception(
                "User is not on the recommendation panel. To choose a service"
                " user has to be on the recommendation panel. Use .go_to_panel"
                " function"
            )
        if k is None:
            new_stage.lluj.click_random_service_from_rec(recommendation)
        else:
            new_stage.lluj.click_service_from_rec(k, recommendation)
        return new_stage
