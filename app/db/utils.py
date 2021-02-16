# pylint: disable-all

import os
from mongoengine import connect, disconnect


def switch_to_main_db():
    disconnect()
    pymongo_client = connect(
        db=os.environ.get("MONGO_DB_NAME"), host=os.environ.get("MONGO_DB_ENDPOINT")
    )

    return pymongo_client


def switch_to_test_db():
    disconnect()
    pymongo_client = connect(
        db=os.environ.get("MONGO_DB_TEST_NAME"),
        host=os.environ.get("MONGO_DB_TEST_ENDPOINT"),
    )

    return pymongo_client
