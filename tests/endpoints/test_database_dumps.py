# pylint: disable-all

import json

from recommender.models import Service


def test_database_dumps_endpoint(client, mp_dump_data):
    client.post(
        "/database_dumps",
        data=json.dumps(mp_dump_data),
        content_type="application/json",
    )

    assert len(Service.objects) > 0
