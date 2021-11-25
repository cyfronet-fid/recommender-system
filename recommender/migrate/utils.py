# pylint: disable=line-too-long

"""Migration utility functions used in flask CLI commands"""

from importlib import import_module

from definitions import ROOT_DIR, MIGRATIONS_DIR
from recommender.migrate.base_migration import BaseMigration
from recommender.models.migration import Migration


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
        print(f"Missing migration module {import_string}... Aborting!")
        return None

    migration_class = BaseMigration.migration_dict[migration.name]
    return migration_class


def scan_for_migrations():
    """
    Scans for migrations in the MIGRATE_DIR and
    then inserts newly found migrations into the Migration collection.
    """

    print("Scanning for new migrations...")
    names_from_migrations_dir = _get_names_from_migrations_dir()
    names_from_db = [migration.name for migration in Migration.objects()]
    missing_names = set(names_from_migrations_dir) - set(names_from_db)

    if len(missing_names) == 0:
        print("\tFound no new migrations!")
        return False

    for missing_name in missing_names:
        print(
            f"\tInserting new migration {missing_name} into the Migration collection..."
        )
        new_migration_model = Migration(name=missing_name, applied=False)
        new_migration_model.save()

    print("\tNew migrations acknowledged!")
    return True


def apply_migrations():
    """Applies unapplied migrations"""

    print("Applying migrations...")

    unapplied_migrations = Migration.objects(applied=False).order_by("+name")

    if len(unapplied_migrations) == 0:
        print("\tNo unapplied migrations found!")
        return False

    for unapplied in unapplied_migrations:
        print(f"\tApplying `{unapplied.name}` ...")

        # Retrieve class
        migration_class = _get_migration_class(unapplied)

        # Apply migration
        migration_class().up()

        # Update migration document
        unapplied.update(applied=True)

    print("\tDone applying migrations!")
    return True


def rollback_last_migration():
    """Rollbacks last applied migration"""

    print("Rollbacking last applied migration...")

    applied_migrations = Migration.objects(applied=True).order_by("-name")

    if len(applied_migrations) == 0:
        print("\tThere are no applied migrations to rollback!")
        return False

    last_applied = applied_migrations.first()
    print(f"\tRollbacking `{last_applied.name}` ...")

    # Retrieve class
    migration_class = _get_migration_class(last_applied)

    try:
        # Rollback migration
        migration_class().down()
    except NotImplementedError:
        print("f\t Cannot rollback migration, no 'down' method, aborting!")
        return False

    # Update migration document
    last_applied.update(applied=False)

    print("\tDone rollbacking!")
    return True


def list_migrations():
    """Lists migrations along with their application status"""

    print("Listing migrations...")

    migrations = Migration.objects().order_by("+name")

    if len(migrations) == 0:
        print("\tNo migrations found!")
        return False

    for migration in migrations:
        print(
            f"\t {migration.name} ... {'applied' if migration.applied else 'not applied'}"
        )
    return True


def check_migration_integrity():
    """
    Checks if migrations in the Migration collection
    have file representations in the MIGRATE_DIR
    """

    print("Checking migration integrity...")

    names_from_migrations_dir = set(_get_names_from_migrations_dir())
    names_from_db = [migration.name for migration in Migration.objects()]
    passed = True

    for db_migration in names_from_db:
        if db_migration in names_from_migrations_dir:
            print(f"\t{db_migration} ... OK")
        else:
            print(f"\t{db_migration} ... FILE MISSING!")
            passed = False

    print(f"\tIntegrity check {'passed' if passed else 'failed'}!")
    return passed


def purge_migration_collection():
    """Deletes all migration document from the Migration collection"""

    print("Purging the Migration collection...")

    print("\tDeleting all existing migration documents...")
    Migration.objects().delete()

    print("\tPurging complete!")
    return True
