# pylint: disable-all

from pandas import DataFrame

from recommender.engine.pre_agent.preprocessing.mongo_to_dataframe import (
    service_to_df,
    user_to_df,
    create_raw_dataset,
    USERS,
    SERVICES,
    LABELS,
)
from tests.factories.marketplace import ServiceFactory, UserFactory


def test_service_to_df(mongo):
    service = ServiceFactory()
    df = service_to_df(service)
    assert isinstance(df, DataFrame)


def test_user_to_df(mongo):
    user = UserFactory()
    df = user_to_df(user)
    assert isinstance(df, DataFrame)


def test_create_raw_dataset(mongo):
    _no_one_services = [ServiceFactory() for _ in range(20)]
    common_services = [ServiceFactory() for _ in range(5)]

    user1 = UserFactory()
    user1.accessed_services = user1.accessed_services + common_services
    user1.save()

    user2 = UserFactory()
    user2.accessed_services = user2.accessed_services + common_services
    user2.save()

    raw_dataset = create_raw_dataset()

    assert isinstance(raw_dataset[USERS], DataFrame)
    assert len(raw_dataset[USERS]) >= 20

    assert isinstance(raw_dataset[SERVICES], DataFrame)
    assert len(raw_dataset[SERVICES]) >= 20

    assert isinstance(raw_dataset[LABELS], DataFrame)
    assert len(raw_dataset[LABELS]) >= 20
