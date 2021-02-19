# pylint: disable-all

import factory
from .target import TargetFactory
from .root import RootFactory
from recommender.models import Source


class SourceFactory(TargetFactory):
    class Meta:
        model = Source

    root = None

    class Params:
        recommendation_root = factory.Trait(
            root=factory.SubFactory(RootFactory, recommendation_panel=True)
        )
        regular_list_root = factory.Trait(
            root=factory.SubFactory(RootFactory, regular_services_list=True)
        )
