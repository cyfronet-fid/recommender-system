# pylint: disable-all

import json


def test_database_dumps_endpoint(client, mp_dump_data, mocker):
    # Normally we would patch "recommender.api.endpoints.database_dumps.handle_db_data",
    # but database_dumps endpoint imports handle_db_dump - not handle_db_dump.delay,
    # and later in the code it uses uses handle_db_dump.delay so we have to patch the source module.
    # It has to be that way due to how celery tasks are called.
    # https://docs.python.org/3/library/unittest.mock.html#where-to-patch
    delay_dump = mocker.patch("recommender.tasks.db.handle_db_dump.delay")

    client.post(
        "/database_dumps",
        data=json.dumps(mp_dump_data),
        content_type="application/json",
    )

    delay_dump.assert_called_once_with(mp_dump_data)
