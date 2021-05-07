# pylint: disable-all

from factory import SubFactory, LazyFunction
from factory.mongoengine import MongoEngineFactory
from factory.random import reseed_random, random
from faker import Factory as FakerFactory
from faker.providers import BaseProvider

from recommender.models import SearchData, AccessMode, AccessType, Service, LifeCycleStatus, TargetUser, Trl, Category, Platform, \
    Provider, ScientificDomain
from .marketplace.category import CategoryFactory
from .marketplace.faker_seeds.utils.loaders import load_names_descs, load_names, load_taglines
from .marketplace.platform import PlatformFactory
from .marketplace.provider import ProviderFactory
from .marketplace.scientific_domain import ScientificDomainFactory
from .marketplace.target_user import TargetUserFactory

reseed_random("test-seed")
fake = FakerFactory.create()


class SearchDataProvider(BaseProvider):
    def search_phrase(self) -> str:
        # TODO: preloading of corpuses (now they are loaded from the disc each time)
        """
        Generates search phrase based on text data in the database.

        Returns:
            search_phrase: Search phrase text.
        """

        choice = random.randint(0, 2)
        if choice == 0:
            name_desc_classes = [AccessMode, AccessType, LifeCycleStatus, Service, TargetUser, Trl]
            clazz = random.choice(name_desc_classes)

            corpus = load_names_descs(clazz)

            if random.randint(0, 1):
                corpus = corpus.keys()
            else:
                corpus = corpus.values()
        elif choice == 1:
            name_classes = [Category, Platform, Provider, ScientificDomain]
            clazz = random.choice(name_classes)
            corpus = load_names(clazz)
        else:
            corpus = load_taglines()

        sentences = list(filter(lambda x: x is not None, corpus))
        # TODO: empty sentences handling
        sentence = random.choice(sentences)
        search_phrase_len = random.randint(1, 10)
        words = sentence.split()

        if len(words) <= search_phrase_len:
            search_phrase_words = words
        else:
            start_idx = random.randint(0, len(words) - search_phrase_len)
            search_phrase_words = words[start_idx:start_idx + search_phrase_len]
        search_phrase = " ".join(search_phrase_words)

        return search_phrase


fake.add_provider(SearchDataProvider)


class SearchDataFactory(MongoEngineFactory):
    class Meta:
        model = SearchData

    q = LazyFunction(lambda: fake.search_phrase())

    categories = LazyFunction(
        lambda: CategoryFactory.create_batch(random.randint(2, 5))
    )
    geographical_availabilities = LazyFunction(
        lambda: [fake.country_code() for _ in range(random.randint(2, 5))]
    )
    order_type = "open_access"  # TODO: order type factories
    providers = LazyFunction(
        lambda: ProviderFactory.create_batch(random.randint(2, 5))
    )
    related_platforms = LazyFunction(
        lambda: PlatformFactory.create_batch(random.randint(2, 5))
    )
    scientific_domains = LazyFunction(
        lambda: ScientificDomainFactory.create_batch(random.randint(2, 5))
    )
    sort = "_score"  # WARRNING: this field is useless and should be removed
                     # from SearchData
    target_users = LazyFunction(
        lambda: TargetUserFactory.create_batch(random.randint(2, 5))
    )
