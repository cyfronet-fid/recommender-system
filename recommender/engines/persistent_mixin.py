# pylint: disable=line-too-long
"""Persistent Mixin"""

import pickle
from abc import ABC
from typing import Optional

from recommender.errors import NoSavedMLComponentError
from recommender.models.ml_component import MLComponent


class Persistent(ABC):
    """Inherit this Mixin for classes that should be persistently saved into
    MLComponents table in DB
    """

    @classmethod
    def fetch_latest_component(cls, version: str = None) -> Optional[MLComponent]:
        """
        Check whether certain ML model exists in the MLComponents
         and return the newest object from the DB that matches criteria.
        Args:
            version: Any, user-defined string.

        Returns:
            ml_component: Newest ML model that matches criteria.
        """
        if version is None:
            ml_component = (
                MLComponent.objects(type=cls.__name__).order_by("-id").first()
            )
        else:
            ml_component = (
                MLComponent.objects(type=cls.__name__, version=version)
                .order_by("-id")
                .first()
            )

        return ml_component

    @classmethod
    def load(cls, version: str = None) -> Optional[MLComponent]:
        """
        Load object from MLComponents based on the class and version of an object
        Args:
            version: Any, user-defined string.

        Returns:
            loaded_ml_component: Loaded version of the newest ML model that matches criteria.

        """
        ml_component = cls.fetch_latest_component(version)
        if ml_component is None:
            raise NoSavedMLComponentError(
                f"No saved ML component with version {version}!"
            )

        loaded_ml_component = pickle.loads(ml_component.binary_object)

        return loaded_ml_component

    def save(self, version: str) -> None:
        """
        Saves objet that class inherits this mixin into MComponents table
         in DB.

        Args:
            version: Any, user-defined string.

        """
        MLComponent(
            type=self.__class__.__name__,
            version=version,
            binary_object=pickle.dumps(self),
        ).save()
