# pylint: disable-all

from recommender.commands.migrate import _apply, _rollback
from recommender.models.migration import Migration


def test__apply_happy_path(mongo, mock_migrations_dir):
    # Apply migrations twice and see if the state of the Migration cache is consistent
    for _ in range(2):
        _apply()
        migration_objects = Migration.objects().order_by("+name")

        assert len(migration_objects) == 2

        for m in migration_objects:
            assert m.applied


def test__apply_integrity_check_fail(mongo, mock_migrations_dir):
    Migration(name="20211203092545_missing_migration", applied=False).save()

    # Should fail because there is no '20211203092545_missing_migration.py' file in migrations dir
    _apply()

    # Should be as if nothing has changed, hence the assertions
    migration_objects = Migration.objects().order_by("+name")
    assert len(migration_objects) == 3
    for m in migration_objects:
        assert not m.applied


def test__rollback_happy_path(mongo, mock_migrations_dir):
    _apply()

    migration_objects = Migration.objects().order_by("+name")

    assert migration_objects[0].applied
    assert migration_objects[1].applied

    _rollback()
    assert migration_objects[0].applied
    assert not migration_objects[1].applied

    _rollback()
    assert not migration_objects[0].applied
    assert not migration_objects[0].applied

    # Sanity check - no more migrations unapplied migrations to rollback
    _rollback()
    assert not migration_objects[0].applied
    assert not migration_objects[0].applied


def test__rollback_integrity_check_fail(mongo, mock_migrations_dir):
    _apply()

    # Simulating the situation where there is no file equivalent of the migration in the Migration db cache
    Migration(name="20211203092545_missing_migration", applied=True).save()

    _rollback()

    migration_objects = Migration.objects().order_by("+name")

    # Should be as if nothing has changed, hence the assertions.
    # Additional migration is the invalid, missing migration added manually in the previous lines,
    # So the overall migration count is 3
    assert len(migration_objects) == 3
    for m in migration_objects:
        assert m.applied
