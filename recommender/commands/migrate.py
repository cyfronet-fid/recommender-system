# pylint: disable=missing-function-docstring

"""Flask CLI migrate commands"""

from recommender.migrate.utils import (
    scan_for_migrations,
    check_migration_integrity,
    apply_migrations,
    rollback_last_migration,
    list_migrations,
    purge_migration_collection,
)


def _apply():
    """
    Scans for new migration files,
    checks migration integrity and applies all unapplied migrations
    """
    scan_for_migrations()
    if not check_migration_integrity():
        return

    apply_migrations()


def _rollback():
    """
    Scans for new migration files,
    checks migration integrity and rollbacks last unapplied migration
    """
    scan_for_migrations()
    if not check_migration_integrity():
        return

    rollback_last_migration()


def _list():
    """Lists all migrations"""
    scan_for_migrations()
    list_migrations()


def _check():
    """Checks for migrations integrity"""
    check_migration_integrity()


def _repopulate():
    """Purges migration DB cache and scans migration files"""
    purge_migration_collection()
    scan_for_migrations()


def migrate_command(task):
    globals()[f"_{task}"]()
