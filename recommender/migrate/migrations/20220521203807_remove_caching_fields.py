# pylint: disable=invalid-name, missing-module-docstring

from recommender.migrate.base_migration import BaseMigration


class RemoveCachingFields(BaseMigration):
    """Rename `tensor` field to `one_hot_tensor`"""

    def up(self):
        self.pymongo_db.user.update_many(
            {}, {"$unset": {"dataframe": "", "one_hot_tensor": "", "tensor": ""}}
        )
        self.pymongo_db.service.update_many(
            {}, {"$unset": {"dataframe": "", "one_hot_tensor": "", "tensor": ""}}
        )

    def down(self):
        self.pymongo_db.user.update_many(
            {}, {"$set": {"dataframe": "", "one_hot_tensor": "", "tensor": ""}}
        )
        self.pymongo_db.service.update_many(
            {},
            {"$set": {"dataframe": "null", "one_hot_tensor": "null", "tensor": "null"}},
        )
