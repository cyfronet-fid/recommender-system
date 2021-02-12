import json


def mongo_model_to_json(model):
    json_repr = json.loads(model.to_json())
    json_repr["id"] = model.id
    del json_repr["_id"]
    del json_repr["_cls"]
    return json_repr
