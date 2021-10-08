# pylint: disable-all
from copy import deepcopy

from inflection import underscore, pluralize

from tests.helpers import mongo_model_to_json
from recommender.models import MP_DUMP_MODEL_CLASSES
from recommender.services.mp_dump import load_mp_dump, drop_mp_dump


def test_load_and_drop_mp_dump(mongo, mp_dump_data):
    load_mp_dump(mp_dump_data)

    mongo_objects = {
        underscore(pluralize(model.__name__)): list(model.objects.all())
        for model in MP_DUMP_MODEL_CLASSES
    }

    raw_mongo_objects = {
        k: [mongo_model_to_json(x) for x in v] for k, v in mongo_objects.items()
    }

    raw_mongo_objects_no_internal_data = deepcopy(raw_mongo_objects)

    for k, v in raw_mongo_objects_no_internal_data.items():
        if k == "services" or k == "users":
            for mongo_json_repr in v:
                mongo_json_repr.pop("one_hot_tensor", None)
                mongo_json_repr.pop("dense_tensor", None)
                mongo_json_repr.pop("synthetic", None)

    assert raw_mongo_objects_no_internal_data == mp_dump_data

    # Some proper dereference model checks
    assert [
        mongo_model_to_json(x) for x in mongo_objects["services"][0].providers
    ] == raw_mongo_objects["providers"]
    assert [
        mongo_model_to_json(x) for x in mongo_objects["services"][0].categories
    ] == raw_mongo_objects["categories"]
    assert [
        mongo_model_to_json(x) for x in mongo_objects["users"][0].accessed_services
    ] == raw_mongo_objects["services"]

    drop_mp_dump()

    for model in MP_DUMP_MODEL_CLASSES:
        assert model.objects.first() is None
