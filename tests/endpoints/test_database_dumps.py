# pylint: disable-all

import json


def test_database_dumps_endpoint(client, mp_dump_data):
    client.post(
        "/database_dumps",
        data=json.dumps(mp_dump_data),
        content_type="application/json",
    )

    # assert len(Service.objects) > 0
    # TODO: implement celery into test environment
