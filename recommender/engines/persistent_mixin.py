"""Persistent Mixin"""

import pickle
from abc import ABC
from typing import Any

from recommender.errors import NoSavedMLComponentError
from recommender.models.ml_component import MLComponent
from logger_config import get_logger

logger = get_logger(__name__)


class Persistent(ABC):
    """Inherit this Mixin for classes that should be persistently saved into
    MLComponents table in DB
    """

    @classmethod
    def load(cls, version: str = None) -> Any:
        """
        Load object from MLComponents. It will look for object with type of the
         class that inherit this mixin and with provided version. It will
         return the newest one object in the DB that matches criteria.
        Args:
            version: Any, user-defined string.

        Returns:
            ml_component: Newest object that matches criteria.

        """
        if version is None:
            last_object = MLComponent.objects(type=cls.__name__).order_by("-id").first()
        else:
            last_object = (
                MLComponent.objects(type=cls.__name__, version=version)
                .order_by("-id")
                .first()
            )

        if last_object is None:
            raise NoSavedMLComponentError(
                f"No saved ML component with version {version}!"
            )

        ml_component = pickle.loads(last_object.binary_object)

        return ml_component

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
