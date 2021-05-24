# pylint: disable-all

import pytest

from recommender.models import Category, ScientificDomain, User, ScikitLearnTransformer
from recommender.services.synthetic_dataset.users import (
    _filter_relevant,
    _sample_niche,
    _synthesize_user,
    synthesize_users,
    SYNTHETIC_USERS,
)
from tests.factories.marketplace import ServiceFactory, UserFactory
from tests.factories.marketplace.category import CategoryFactory
from tests.factories.marketplace.scientific_domain import ScientificDomainFactory


@pytest.fixture
def categories(mongo):
    return [CategoryFactory(name=f"c{i}") for i in range(5)]


@pytest.fixture
def scientific_domains(mongo):
    return [ScientificDomainFactory(name=f"sd{i}") for i in range(5)]


@pytest.fixture
def clusters():
    return ["cluster1", "cluster2", "cluster3"]


@pytest.fixture
def category_clusters(categories, clusters):
    return {
        categories[0].name: clusters[:2],
        categories[1].name: clusters,
        categories[2].name: [clusters[1]],
        categories[3].name: clusters,
        categories[4].name: clusters[:2],
    }


@pytest.fixture
def scientific_domain_clusters(scientific_domains, clusters):
    return {
        scientific_domains[0].name: clusters[:2],
        scientific_domains[1].name: clusters,
        scientific_domains[2].name: [clusters[1]],
        scientific_domains[3].name: clusters,
        scientific_domains[4].name: clusters[:2],
    }


@pytest.fixture
def services(categories, scientific_domains):
    return [
        ServiceFactory(
            categories=categories[:2], scientific_domains=scientific_domains[:2]
        ),
        ServiceFactory(
            categories=[categories[2]], scientific_domains=[scientific_domains[4]]
        ),
    ]


def test__filter_relevant(services, categories, scientific_domains):
    assert set(_filter_relevant(Category)) == set(map(lambda x: x.name, categories[:3]))
    assert set(_filter_relevant(ScientificDomain)) == set(
        map(lambda x: x.name, scientific_domains[:2] + [scientific_domains[4]])
    )


def test__sample_niche(clusters, category_clusters, categories):
    assert len(_sample_niche(clusters[0], category_clusters, sample_range=(1, 3))) >= 1
    assert len(_sample_niche(clusters[0], category_clusters, sample_range=(2, 2))) == 2
    assert (
        len(
            _sample_niche(
                clusters[0],
                category_clusters,
                sample_range=(len(categories) + 1, len(categories) + 1),
            )
        )
        == 4
    )
    assert set(
        _sample_niche(clusters[0], category_clusters, sample_range=(4, 4))
    ) == set(c.name for c in categories[:2] + categories[3:])
    assert set(
        _sample_niche(clusters[1], category_clusters, sample_range=(5, 5))
    ) == set(c.name for c in categories)


def test__synthesize_user(
    clusters,
    category_clusters,
    categories,
    scientific_domain_clusters,
    scientific_domains,
):
    _synthesize_user(
        clusters[0], category_clusters, scientific_domain_clusters, niche_range=(4, 4)
    )
    assert len(User.objects) == 1
    assert User.objects.first().id == 0
    assert User.objects.first().synthetic
    assert len(User.objects.first().accessed_services) == 0
    assert User.objects.first().categories == categories[:2] + categories[3:]
    assert (
        User.objects.first().scientific_domains
        == scientific_domains[:2] + scientific_domains[3:]
    )

    _synthesize_user(
        clusters[1], category_clusters, scientific_domain_clusters, niche_range=(5, 5)
    )
    assert len(User.objects) == 2
    assert User.objects[1].id == 1
    assert User.objects[1].synthetic
    assert len(User.objects[1].accessed_services) == 0
    assert User.objects[1].categories == categories
    assert User.objects[1].scientific_domains == scientific_domains


def test__synthesize_user_with_real_users_in_db(
    clusters,
    category_clusters,
    categories,
    scientific_domain_clusters,
    scientific_domains,
):

    real_user = UserFactory()
    _synthesize_user(
        clusters[0], category_clusters, scientific_domain_clusters, niche_range=(4, 4)
    )

    assert len(User.objects) == 2
    assert User.objects[1].id == User.objects[0].id + 1
    assert not User.objects[0].synthetic
    assert User.objects[1].synthetic

    assert len(User.objects[1].accessed_services) == 0
    assert User.objects[1].categories == categories[:2] + categories[3:]
    assert (
        User.objects[1].scientific_domains
        == scientific_domains[:2] + scientific_domains[3:]
    )


def test_synthesize_users(mongo):
    synthesize_users(100)
    assert len(User.objects) == 100

    assert User.objects.first().dataframe
    assert len(ScikitLearnTransformer.objects) == 1
    assert ScikitLearnTransformer.objects.first().name == SYNTHETIC_USERS
