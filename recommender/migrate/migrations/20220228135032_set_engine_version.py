# pylint: disable=invalid-name, missing-module-docstring, line-too-long

from recommender.migrate.base_migration import BaseMigration


class SetEngineVersion(BaseMigration):
    """
    Set engine_version parameter to "NCF" for every recommendation object which does not have this parameter set.
    All recommendations were handled by the NCF algorithm prior to the introduction of RL as the recommendation-making algorithm.
    """

    def up(self):
        self.pymongo_db.recommendation.update_many(
            {"engine_version": {"$eq": None}}, {"$set": {"engine_version": "NCF"}}
        )

    def down(self):
        # Information about which recommendation objects exactly did not have the engine_version field set is lost.
        pass
