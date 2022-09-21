# pylint: disable=invalid-name, missing-module-docstring

from recommender.migrate.base_migration import BaseMigration


class RenameElasticServicesToCandidates(BaseMigration):
    """Rename `elastic_services` field to `candidates`"""

    def up(self):
        self.pymongo_db.recommendation.update_many(
            {}, {"$rename": {"elastic_services": "candidates"}}
        )
        self.pymongo_db.state.update_many(
            {}, {"$rename": {"elastic_services": "candidates"}}
        )

    def down(self):
        self.pymongo_db.recommendation.update_many(
            {}, {"$rename": {"candidates": "elastic_services"}}
        )
        self.pymongo_db.state.update_many(
            {}, {"$rename": {"candidates": "elastic_services"}}
        )
