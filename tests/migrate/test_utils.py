# pylint: disable-all
from importlib import import_module

import pytest

from recommender.migrate.utils import (
    scan_for_migrations,
    apply_migrations,
    rollback_last_migration,
    check_migration_integrity,
    purge_migration_collection,
    _get_migration_class,
)
from recommender.models.migration import Migration


@pytest.fixture
def mock_migration_objects():
    return [
        Migration(name="20211203084612_first_mock_migration", applied=False).save(),
        Migration(name="20211203092545_second_mock_migration", applied=False).save(),
    ]


@pytest.fixture()
def migration_method_mocks(mocker):
    first_up = mocker.patch(
        "tests.migrate.mock_migrations.20211203084612_first_mock_migration.FirstMigrationForTesting.up"
    )
    second_up = mocker.patch(
        "tests.migrate.mock_migrations.20211203092545_second_mock_migration.SecondMigrationForTesting.up"
    )

    first_down = mocker.patch(
        "tests.migrate.mock_migrations.20211203084612_first_mock_migration.FirstMigrationForTesting.down"
    )
    second_down = mocker.patch(
        "tests.migrate.mock_migrations.20211203092545_second_mock_migration.SecondMigrationForTesting.down"
    )

    return [first_up, second_up], [first_down, second_down]


def test_scan_for_migrations(mongo, mock_migrations_dir):
    scan_for_migrations()

    migration_objects = Migration.objects().order_by("+name")
    assert len(migration_objects) == 2

    first_migration = migration_objects[0]
    assert first_migration.name == "20211203084612_first_mock_migration"
    assert not first_migration.applied

    second_migration = migration_objects[1]
    assert second_migration.name == "20211203092545_second_mock_migration"
    assert not second_migration.applied


def test_apply_migrations(
    mongo, migration_method_mocks, mock_migrations_dir, mock_migration_objects
):
    apply_migrations()

    for m in mock_migration_objects:
        m.reload()
        assert m.applied

    up_mocks, down_mocks = migration_method_mocks

    for up in up_mocks:
        up.assert_called_once()

    for down in down_mocks:
        assert not down.called


def test_rollback_last_migration(
    mongo, migration_method_mocks, mock_migrations_dir, mock_migration_objects
):
    for m in mock_migration_objects:
        m.update(applied=True)

    rollback_last_migration()

    last_migration = mock_migration_objects[-1]
    last_migration.reload()
    assert not last_migration.applied

    up_mocks, down_mocks = migration_method_mocks

    for up in up_mocks:
        assert not up.called

    assert not down_mocks[0].called
    down_mocks[1].assert_called_once()


def test_check_migration_integrity(mongo, mock_migration_objects, mock_migrations_dir):
    assert check_migration_integrity()
    Migration(name="20211203092545_missing_migration", applied=False).save()
    assert not check_migration_integrity()


def test_purge_migration_collection(mongo, mock_migration_objects):
    purge_migration_collection()
    assert len(Migration.objects()) == 0


def test__get_migration_class(mongo, mock_migration_objects, mock_migrations_dir):
    first_migration_module = import_module(
        "tests.migrate.mock_migrations.20211203084612_first_mock_migration"
    )
    first_migration_class = getattr(first_migration_module, "FirstMigrationForTesting")

    assert _get_migration_class(mock_migration_objects[0]) == first_migration_class
