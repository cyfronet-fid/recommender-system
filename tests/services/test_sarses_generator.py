# pylint: disable-all

from tests.factories.marketplace import UserFactory
from tests.factories.recommendation import RecommendationFactory, faker
from tests.factories.user_action import UserActionFactory
from recommender.engines.rl.ml_components.real_sarses import generate_real_sarses
from recommender.models import Sars, UserAction
from recommender.engines.rl.ml_components.real_sarses.reward_mapping import (
    ua_to_reward_id,
)
from recommender.engines.rl.ml_components.real_sarses import (
    RECOMMENDATION_PAGES_IDS,
    _tree_collapse,
    _get_clicked_services_and_reward,
    _find_root_uas_before,
)
from faker import Factory as FakerFactory


def ruas2services(ruas):
    return [rua.source.root.service for rua in ruas]


class TestSarsesGenerator:
    def test_tree_colapse(self, mongo):
        # Zero level
        root_ua = UserActionFactory(recommendation_root=True)

        rewards = _tree_collapse(root_ua)
        assert type(rewards) == list
        assert len(rewards) == 1

        # First level
        ua0, ua1, ua2, ua3 = UserActionFactory.create_batch(
            4, source__visit_id=root_ua.target.visit_id
        )

        rewards = _tree_collapse(root_ua)
        assert type(rewards) == list
        assert len(rewards) == 5

        # Second level
        ua4 = UserActionFactory(source__visit_id=ua1.target.visit_id)

        ua5 = UserActionFactory(source__visit_id=ua2.target.visit_id)
        ua5.target.page_id = RECOMMENDATION_PAGES_IDS[0]
        ua5.save()
        ua6 = UserActionFactory(source__visit_id=ua2.source.visit_id)

        ua7, ua8, ua9 = UserActionFactory.create_batch(
            3, source__visit_id=ua3.target.visit_id
        )
        ua9.action.order = True
        ua9.save()

        rewards = _tree_collapse(root_ua)
        assert type(rewards) == list
        assert len(rewards) == 11

        # Third level
        ua10 = UserActionFactory(source__visit_id=ua5.target.visit_id)

        ua11 = UserActionFactory(source__visit_id=ua8.target.visit_id)
        ua11.target.page_id = RECOMMENDATION_PAGES_IDS[0]
        ua11.save()
        ua12 = UserActionFactory(source__visit_id=ua8.target.visit_id)

        ua13, ua14, ua15 = UserActionFactory.create_batch(
            3, source__visit_id=ua9.target.visit_id
        )

        rewards = _tree_collapse(root_ua)
        assert type(rewards) == list
        assert len(rewards) == 13

    def test_get_clicked_services_and_reward(self, mongo):
        [UserActionFactory(recommendation_root=True) for _ in range(3)]

        recommendation1 = RecommendationFactory(v1=True)
        [UserActionFactory(recommendation_root=True) for _ in range(3)]
        root_uas = UserAction.objects(source__root__type__="recommendation_panel")

        clicked_services_after, reward = _get_clicked_services_and_reward(
            recommendation1, root_uas
        )

        assert type(clicked_services_after) == list
        assert len(clicked_services_after) == 0

        for service_reward in reward:
            assert type(service_reward) == list
            assert len(service_reward) == 0

        recommendation2 = RecommendationFactory(v1=True)
        UserActionFactory(
            recommendation_root=True,
            source__visit_id=recommendation2.visit_id,
            source__root__service=recommendation2.services[0],
        )

        clicked_services_after, reward = _get_clicked_services_and_reward(
            recommendation2, root_uas
        )

        assert type(clicked_services_after) == list
        assert len(clicked_services_after) == 1

        for service_reward in reward:
            assert type(service_reward) == list

        assert len(reward[0]) == 1
        assert len(reward[1]) == 0
        assert len(reward[2]) == 0

        UserActionFactory(
            recommendation_root=True,
            source__visit_id=recommendation2.visit_id,
            source__root__service=recommendation2.services[1],
        )

        clicked_services_after, reward = _get_clicked_services_and_reward(
            recommendation2, root_uas
        )

        assert type(clicked_services_after) == list
        assert len(clicked_services_after) == 2

        for service_reward in reward:
            assert type(service_reward) == list

        assert len(reward[0]) == 1
        assert len(reward[1]) == 1
        assert len(reward[2]) == 0

        UserActionFactory(
            recommendation_root=True,
            source__visit_id=recommendation2.visit_id,
            source__root__service=recommendation2.services[1],
        )

        clicked_services_after, reward = _get_clicked_services_and_reward(
            recommendation2, root_uas
        )

        assert type(clicked_services_after) == list
        assert len(clicked_services_after) == 2

        for service_reward in reward:
            assert type(service_reward) == list

        assert len(reward[0]) == 1
        assert len(reward[1]) == 2
        assert len(reward[2]) == 0

    def test_find_root_uas_before(self, mongo):
        user = UserFactory()
        root_uas = [
            UserActionFactory(recommendation_root=True, user=user) for _ in range(3)
        ]
        found_root_uas = UserAction.objects(source__root__type__="recommendation_panel")
        recommendation = RecommendationFactory(user=user)

        assert root_uas == list(_find_root_uas_before(found_root_uas, recommendation))

        unique_id = FakerFactory.create().uuid4()

        root_uas = [
            UserActionFactory(
                recommendation_root=True, not_logged=True, unique_id=unique_id
            )
            for _ in range(3)
        ]
        found_root_uas = UserAction.objects(
            source__root__type__="recommendation_panel", unique_id=unique_id
        )
        recommendation = RecommendationFactory(not_logged=True, unique_id=unique_id)

        assert root_uas == list(_find_root_uas_before(found_root_uas, recommendation))

    def test_generate_sarses(self, mongo):
        user = UserFactory()

        # User's root actions taken before considered recommendation
        root_actions_before_recommendation = [
            UserActionFactory(recommendation_root=True, user=user) for _ in range(3)
        ]

        # Simple user journey
        recommendation = RecommendationFactory(v1=True, user=user)

        root_user_action_1 = UserActionFactory(
            recommendation_root=True,
            user=user,
            source__visit_id=recommendation.visit_id,
            source__root__service=recommendation.services[0],
        )

        root_user_action_2 = UserActionFactory(
            recommendation_root=True,
            user=user,
            source__visit_id=recommendation.visit_id,
            source__root__service=recommendation.services[2],
        )

        non_root_user_action = UserActionFactory(
            user=user,
            source__visit_id=root_user_action_2.target.visit_id,
        )

        _next_recommendation = RecommendationFactory(v1=True, user=user)

        generate_real_sarses(multi_processing=False)
        sars = Sars.objects.first()

        clicked_before = user.accessed_services + ruas2services(
            root_actions_before_recommendation
        )
        assert sars.state.services_history == clicked_before

        clicked_after = [
            root_user_action_1.source.root.service,
            root_user_action_2.source.root.service,
        ]
        assert sars.action == recommendation.services

        assert sars.reward == [
            [ua_to_reward_id(root_user_action_1)],
            [],
            [
                ua_to_reward_id(root_user_action_2),
                ua_to_reward_id(non_root_user_action),
            ],
        ]
        assert sars.next_state.services_history == clicked_before + clicked_after

    def test_generate_sarses_anonymous(
        self,
        mongo,
    ):
        # At least one user in the DB assumption, needed for _get_empty_user()
        UserFactory()

        unique_id = faker.uuid4()

        # Anonymous user's root actions taken before considered recommendation
        root_actions_before_recommendation = [
            UserActionFactory(
                recommendation_root=True, not_logged=True, unique_id=unique_id
            )
            for _ in range(3)
        ]

        # Simple user journey
        recommendation = RecommendationFactory(
            v1=True, not_logged=True, unique_id=unique_id
        )

        root_user_action_1 = UserActionFactory(
            recommendation_root=True,
            not_logged=True,
            unique_id=unique_id,
            source__visit_id=recommendation.visit_id,
            source__root__service=recommendation.services[0],
        )

        root_user_action_2 = UserActionFactory(
            recommendation_root=True,
            not_logged=True,
            unique_id=unique_id,
            source__visit_id=recommendation.visit_id,
            source__root__service=recommendation.services[2],
        )

        non_root_user_action = UserActionFactory(
            not_logged=True,
            unique_id=unique_id,
            source__visit_id=root_user_action_2.target.visit_id,
        )

        _next_recommendation = RecommendationFactory(
            v1=True, not_logged=True, unique_id=unique_id
        )

        generate_real_sarses(multi_processing=False)
        sars = Sars.objects.first()

        clicked_before = ruas2services(root_actions_before_recommendation)
        assert sars.state.services_history == clicked_before

        clicked_after = [
            root_user_action_1.source.root.service,
            root_user_action_2.source.root.service,
        ]
        assert sars.action == recommendation.services

        assert sars.reward == [
            [ua_to_reward_id(root_user_action_1)],
            [],
            [
                ua_to_reward_id(root_user_action_2),
                ua_to_reward_id(non_root_user_action),
            ],
        ]
        assert sars.next_state.services_history == clicked_before + clicked_after
