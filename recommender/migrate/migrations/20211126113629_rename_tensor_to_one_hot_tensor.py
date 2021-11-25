# pylint: disable=invalid-name, missing-module-docstring

from recommender.migrate.base_migration import BaseMigration


class RenameTensorToOneHotTensor(BaseMigration):
    """Rename `tensor` field to `one_hot_tensor`"""

    def up(self):
        self.pymongo_db.user.update_many({}, {"$rename": {"tensor": "one_hot_tensor"}})
        self.pymongo_db.service.update_many(
            {}, {"$rename": {"tensor": "one_hot_tensor"}}
        )

    def down(self):
        self.pymongo_db.user.update_many({}, {"$rename": {"one_hot_tensor": "tensor"}})
        self.pymongo_db.service.update_many(
            {}, {"$rename": {"one_hot_tensor": "tensor"}}
        )
