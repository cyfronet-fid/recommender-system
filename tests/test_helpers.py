def pymodm_json_repr(model):
    json_repr = model.to_son()
    json_repr["id"] = model.id
    del json_repr["_id"]
    del json_repr["_cls"]
    return json_repr
