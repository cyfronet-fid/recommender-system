# pylint: disable=invalid-name

"""Base migration class"""

from abc import ABC, abstractmethod

from mongoengine import get_db


class BaseMigration(ABC):
    """Interface for implementing migrations"""

    migration_dict = dict()

    def __init__(self):
        """Prepares the pymongo adapter for migration"""
        self.pymongo_db = get_db()

    def __init_subclass__(cls):
        name = cls.__module__.rsplit(".", maxsplit=1)[-1]
        BaseMigration.migration_dict[name] = cls

    @abstractmethod
    def up(self):
        """Migration code"""
        raise NotImplementedError

    @abstractmethod
    def down(self):
        """
        Rollback code. Can be left as `pass` in cases where
        it is impossible to rollback the migration (like dropping a collection)
        """
        raise NotImplementedError
