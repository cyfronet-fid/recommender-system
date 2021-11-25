# pylint: disable-all

from recommender.migrate.base_migration import BaseMigration


class FirstMigrationForTesting(BaseMigration):
    def up(self):
        pass

    def down(self):
        pass
