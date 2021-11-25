# pylint: disable-all

from importlib import import_module

from recommender.migrate.base_migration import BaseMigration


def test_migrations_dict(mock_migrations_dir):
    first_migration_module = import_module(
        "tests.migrate.mock_migrations.20211203084612_first_mock_migration"
    )
    second_migration_module = import_module(
        "tests.migrate.mock_migrations.20211203092545_second_mock_migration"
    )

    first_migration_class = getattr(first_migration_module, "FirstMigrationForTesting")
    second_migration_class = getattr(
        second_migration_module, "SecondMigrationForTesting"
    )

    proper_migrations_dict = {
        "20211203084612_first_mock_migration": first_migration_class,
        "20211203092545_second_mock_migration": second_migration_class,
    }

    assert BaseMigration.migration_dict == proper_migrations_dict


def test_pymongo_db_var(mongo, mock_migrations_dir):
    first_migration_module = import_module(
        "tests.migrate.mock_migrations.20211203084612_first_mock_migration"
    )
    first_migration_class = getattr(first_migration_module, "FirstMigrationForTesting")

    migration = first_migration_class()

    assert migration.pymongo_db.client.address == ("localhost", 27017)
    assert migration.pymongo_db.name == "test"
