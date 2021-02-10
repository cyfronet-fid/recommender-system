"""Helper util methods for tests"""

import json


def mongo_model_to_json(model):
    """Convert mongoeninge model to json representation
    that matches our MP serialization"""
    json_repr = json.loads(model.to_json())
    json_repr["id"] = model.id
    del json_repr["_id"]
    del json_repr["_cls"]
    return json_repr
