# pylint: disable-all
from factory import DictFactory


class UserActionFactory(DictFactory):
    user_id = 1234
    unique_id = "5642c351-80fe-44cf-b606-304f2f338122"
    timestamp = "2021-03-18T19:33:21.620Z"
    source = {
        "visit_id": "202090a4-de4c-4230-acba-6e2931d9e37c",
        "page_id": "services_catalogue_list",
        "root": {
            "type": "recommendation_panel",
            "panel_id": "v1",
            "service_id": 1234,
        },
    }

    target = {
        "visit_id": "9f543b80-dd5b-409b-a619-6312a0b04f4f",
        "page_id": "service_about",
    }

    action = {"type": "button", "text": "Details", "order": True}
