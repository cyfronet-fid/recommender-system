# pylint: disable-all
import random

from dotenv import load_dotenv

load_dotenv()
import pytest
from tqdm.auto import trange

from recommender import User
from recommender.engines.panel_id_to_services_number_mapping import PANEL_ID_TO_K
from recommender.utils import visualize_uas
from tests.services.user_journey import UserJourney
from tests.factories.marketplace import UserFactory
from tests.factories.recommendation import RecommendationFactory, faker
from tests.factories.user_action import UserActionFactory
from recommender.engines.rl.ml_components.sarses_generator import (
    generate_sarses,
    regenerate_sarses,
)
from recommender.models import Sars, UserAction, Recommendation, Service
from recommender.engines.rl.ml_components.reward_mapping import ua_to_reward_id
from recommender.engines.rl.ml_components.sarses_generator import (
    RECOMMENDATION_PAGES_IDS,
    _tree_collapse,
    _get_clicked_services_and_reward,
    _find_root_uas_before,
)
from faker import Factory as FakerFactory


def ruas2services(ruas):
    return [rua.source.root.service for rua in ruas]


class TestGenerateSarses:
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

        generate_sarses(multi_processing=False)
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

        generate_sarses(multi_processing=False)
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


class TestRegenerateSarses:
    @pytest.mark.parametrize("multi_processing", [True, False])
    def test_zero_case(self, mongo, multi_processing):
        regenerate_sarses(multi_processing=multi_processing)
        assert Sars.objects.count() == 0

    @pytest.mark.parametrize("multi_processing", [True, False])
    @pytest.mark.parametrize("collection", [User, Service, Sars])
    def test_missing_data_case(self, mongo, multi_processing, collection):
        j = UserJourney()
        j.go_to_panel()

        regenerate_sarses(multi_processing=multi_processing)
        assert Sars.objects.count() == 1
        original_sars_id = Sars.objects.first().id

        j.next()
        collection.drop_collection()
        regenerate_sarses(multi_processing=multi_processing)

        assert Sars.objects.count() == 0 or (
            Sars.objects.count() == 1 and Sars.objects.first().id != original_sars_id
        )

    @pytest.mark.parametrize("anonymous", [True, False])
    @pytest.mark.parametrize("multi_processing", [True, False])
    @pytest.mark.parametrize("panel_id", list(PANEL_ID_TO_K.keys()))
    def test_minimal_case(self, mongo, anonymous, multi_processing, panel_id):
        UserFactory()  # User necessary for assertion in the `_get_empty_user` function
        UserJourney(anonymous=anonymous, panel_id=panel_id).go_to_panel()
        # visualize_uas(save=True) # Uncomment for visual debugging

        regenerate_sarses(multi_processing=multi_processing)
        assert Sars.objects.count() == 1

    @pytest.mark.parametrize("anonymous", [True, False])
    @pytest.mark.parametrize("multi_processing", [True, False])
    @pytest.mark.parametrize("panel_id", list(PANEL_ID_TO_K.keys()))
    def test_case_with_uas(self, mongo, anonymous, multi_processing, panel_id):
        UserFactory()  # User necessary for assertion in the `_get_empty_user` function

        j = UserJourney(anonymous=anonymous, panel_id=panel_id)
        recommendation = Recommendation.objects(visit_id=j.lluj.last_visit_id).first()
        for k in range(len(recommendation.services)):
            j2 = j.service(k)
        j2.go_to_panel()
        regenerate_sarses(multi_processing=multi_processing)

        assert Sars.objects.count() == 1

    @pytest.mark.parametrize("anonymous", [True, False])
    @pytest.mark.parametrize("multi_processing", [True, False])
    @pytest.mark.parametrize("panel_id", list(PANEL_ID_TO_K.keys()))
    def test_uas_based_regeneration_scenario(
        self, mongo, anonymous, multi_processing, panel_id
    ):
        # Check if sars is regenerated if there are new user actions for the sars's recommendation
        UserFactory()  # User necessary for assertion in the `_get_empty_user` function

        j = UserJourney(anonymous=anonymous, panel_id=panel_id)
        recommendation = Recommendation.objects(visit_id=j.lluj.last_visit_id).first()
        for k in range(len(recommendation.services)):
            j2 = j.service(k)
        j2.go_to_panel()

        # Check if regeneration change `processed` status of recommendations and user actions
        assert all([rec.processed in [False, None] for rec in Recommendation.objects])
        assert all([ua.processed in [False, None] for ua in UserAction.objects])
        regenerate_sarses(multi_processing=multi_processing)
        assert all([rec.processed for rec in Recommendation.objects])
        assert all([ua.processed is True for ua in UserAction.objects])

        # Create a chain of user actions that is rooted in the recommendation
        for _ in range(3):
            original_sars_id = (
                Sars.objects(source_recommendation=recommendation).first().id
            )
            j = j.next()

            # Check if chained user action is processed
            ua = UserAction.objects(target__visit_id=j.lluj.last_visit_id).first()
            assert ua.processed in [False, None]
            regenerate_sarses(multi_processing=multi_processing)
            assert ua.reload().processed is True

            regenerated_sars_id = (
                Sars.objects(source_recommendation=recommendation).first().id
            )

            # visualize_uas(save=True)  # Uncomment for visual debugging

            # Check if sars is regenerated
            assert original_sars_id != regenerated_sars_id
            assert Sars.objects.count() == 1

    @pytest.mark.parametrize("anonymous", [True, False])
    @pytest.mark.parametrize("multi_processing", [True, False])
    @pytest.mark.parametrize("panel_id", list(PANEL_ID_TO_K.keys()))
    def test_multiple_users_and_recommendations_scenario(
        self, mongo, anonymous, multi_processing, panel_id
    ):
        UserFactory()  # User necessary for assertion in the `_get_empty_user` function

        j1 = UserJourney(anonymous=anonymous, panel_id=panel_id)
        source_rec_j1 = Recommendation.objects(visit_id=j1.lluj.last_visit_id).first()
        j1.service(0).next(2).order().go_to_panel()

        j2 = UserJourney(anonymous=anonymous, panel_id=panel_id)
        source_rec_j2 = Recommendation.objects(visit_id=j2.lluj.last_visit_id).first()
        j2_1 = j2.service(0)
        j2_1.next(3).order().go_to_panel()
        j2_1.next(2)
        # visualize_uas(save=True) # Uncomment for visual debugging

        regenerate_sarses(multi_processing=multi_processing)

        # After first regeneration there should be two sarses
        assert Sars.objects.count() == 2
        original_sars_j1_id = (
            Sars.objects(source_recommendation=source_rec_j1).first().id
        )
        original_sars_j2_id = (
            Sars.objects(source_recommendation=source_rec_j2).first().id
        )

        j1_1 = j1.service(1).next(2)
        j1_1_1 = j1_1.order()
        j1_1.next(3)
        j1_1.next(1)

        j2.service(1).next().order()
        # visualize_uas(save=True) # Uncomment for visual debugging

        regenerate_sarses(multi_processing=multi_processing)
        regenerated_sars_j1_id = (
            Sars.objects(source_recommendation=source_rec_j1).first().id
        )
        regenerated_sars_j2_id = (
            Sars.objects(source_recommendation=source_rec_j2).first().id
        )

        # After second regeneration all sarses should be regenerated due to new user actions
        assert Sars.objects.count() == 2
        assert original_sars_j1_id != regenerated_sars_j1_id
        assert original_sars_j2_id != regenerated_sars_j2_id

        j1_1_1.go_to_panel()
        # visualize_uas(save=True) # Uncomment for visual debugging

        regenerate_sarses(multi_processing=multi_processing)
        regenerated_sars_j1_after_id = (
            Sars.objects(source_recommendation=source_rec_j1).first().id
        )
        regenerated_sars_j2_after_id = (
            Sars.objects(source_recommendation=source_rec_j2).first().id
        )

        # After third regeneration only sars related to the first recommendation should be regenerated
        # because only in it's recommendation's user actions tree there are new user actions.
        assert regenerated_sars_j1_id != regenerated_sars_j1_after_id
        assert regenerated_sars_j2_id == regenerated_sars_j2_after_id

    @pytest.mark.skip(
        reason="Needs significant amount of time (7min 21s on 12core i7 machine). Un-skip if needed"
    )
    @pytest.mark.parametrize("multi_processing", [True, False])
    def test_heavy_load_scenario(self, mongo, multi_processing):
        # Integration test that check if regenerate_sarses is not failing on
        # complex users journeys scenario with multiple
        # anonymous and logged users
        for i in trange(10):
            if i % 2 == 0:
                anonymous = True
            else:
                anonymous = False
            j = UserJourney(anonymous=anonymous)
            r = Recommendation.objects(visit_id=j.lluj.last_visit_id).first()
            for _ in trange(5, leave=False):
                ua = j.lluj.click_random_ua_from_rec_tree(r, n=6)
                j.lluj.go_to_rec_page(ua)
                js = [j.service() for _ in range(3)]
                j2 = js[0].next(random.randint(1, 4)).order().go_to_panel()
                j3 = js[1].next(random.randint(1, 4)).go_to_panel()
                j.next(1).order().go_to_panel()
                j.next(2).order().next(1).go_to_panel()
                j.next(3).order().next(1)
            j.lluj.go_to_rec_page(start_ua=ua)

        regenerate_sarses(multi_processing=multi_processing, verbose=False)
