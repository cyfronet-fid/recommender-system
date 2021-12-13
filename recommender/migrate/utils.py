# pylint: disable=line-too-long, fixme

"""Migration utility functions used in flask CLI commands"""

from importlib import import_module

from definitions import ROOT_DIR, MIGRATIONS_DIR
from recommender.migrate.base_migration import BaseMigration
from recommender.models.migration import Migration
from logger_config import get_logger

logger = get_logger(__name__)


def _get_names_from_migrations_dir():
    migration_files = MIGRATIONS_DIR.glob("[0-9]*.py")
    migration_names = [m.stem for m in migration_files]
    return migration_names


def _get_migration_class(migration: Migration):
    relative_migrate_dir = MIGRATIONS_DIR.relative_to(ROOT_DIR)
    migration_module = str(relative_migrate_dir).replace("/", ".")

    # Importing to register the subclass in the
    # BaseMigration "migrations_dict" class variable
    import_string = f"{migration_module}.{migration.name}"
    try:
        import_module(import_string)
    except ModuleNotFoundError:
        logger.error("Missing migration module %s... Aborting!", import_string)
        return None

    migration_class = BaseMigration.migration_dict[migration.name]
    return migration_class


def scan_for_migrations():
    """
    Scans for migrations in the MIGRATE_DIR and
    then inserts newly found migrations into the Migration collection.
    """

    logger.info("Scanning for new migrations...")
    names_from_migrations_dir = _get_names_from_migrations_dir()
    names_from_db = [migration.name for migration in Migration.objects()]
    missing_names = set(names_from_migrations_dir) - set(names_from_db)

    if len(missing_names) == 0:
        logger.info("\tFound no new migrations!")
        return False

    for missing_name in missing_names:
        logger.info(
            "\tInserting new migration %s into the Migration collection...",
            missing_name,
        )
        new_migration_model = Migration(name=missing_name, applied=False)
        new_migration_model.save()

    logger.info("\tNew migrations acknowledged!")
    return True


def apply_migrations():
    """Applies unapplied migrations"""

    logger.info("Applying migrations...")

    unapplied_migrations = Migration.objects(applied=False).order_by("+name")

    if len(unapplied_migrations) == 0:
        logger.info("\tNo unapplied migrations found!")
        return False

    for unapplied in unapplied_migrations:
        logger.info("\tApplying `%s` ...", unapplied.name)

        # Retrieve class
        migration_class = _get_migration_class(unapplied)

        # Apply migration
        migration_class().up()

        # Update migration document
        unapplied.update(applied=True)

    logger.info("\tDone applying migrations!")
    return True


def rollback_last_migration():
    """Rollbacks last applied migration"""

    logger.info("Rollbacking last applied migration...")

    applied_migrations = Migration.objects(applied=True).order_by("-name")

    if len(applied_migrations) == 0:
        logger.error("\tThere are no applied migrations to rollback!")
        return False

    last_applied = applied_migrations.first()
    logger.info("\tRollbacking `%s` ...", last_applied.name)

    # Retrieve class
    migration_class = _get_migration_class(last_applied)

    try:
        # Rollback migration
        migration_class().down()
    except NotImplementedError:
        logger.error("\t Cannot rollback migration, no 'down' method, aborting!")
        return False

    # Update migration document
    last_applied.update(applied=False)

    logger.info("\tDone rollbacking!")
    return True


def list_migrations():
    """Lists migrations along with their application status"""

    logger.info("Listing migrations...")

    migrations = Migration.objects().order_by("+name")

    if len(migrations) == 0:
        logger.info("\t No migrations found!")
        return False

    for migration in migrations:
        logger.info(
            "\t %s ... %s",
            migration.name,
            "applied" if migration.applied else "not applied",
        )
    return True


def check_migration_integrity():
    """
    Checks if migrations in the Migration collection
    have file representations in the MIGRATE_DIR
    """

    logger.info("Checking migration integrity...")

    names_from_migrations_dir = set(_get_names_from_migrations_dir())
    names_from_db = [migration.name for migration in Migration.objects()]
    passed = True

    for db_migration in names_from_db:
        if db_migration in names_from_migrations_dir:
            logger.info("\t%s ... OK", db_migration)
        else:
            logger.error("\t%s ... FILE MISSING!", db_migration)
            passed = False

    logger.info("\tIntegrity check %s!", "passed" if passed else "failed")

    return passed


def purge_migration_collection():
    """Deletes all migration document from the Migration collection"""

    logger.info("Purging the Migration collection...")

    logger.info("\tDeleting all existing migration documents...")
    Migration.objects().delete()

    logger.info("\tPurging complete!")
    return True
