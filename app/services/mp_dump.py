from inflection import underscore, pluralize
from app.db.mongo_models import MP_DUMP_MODEL_CLASSES


def load_mp_dump(data):
    for model in MP_DUMP_MODEL_CLASSES:
        json_key = pluralize(underscore(model.__name__))
        for json_dict in data[json_key]:
            model(**json_dict).save()


def drop_mp_dump():
    for model in MP_DUMP_MODEL_CLASSES:
        for obj in model.objects:
            obj.delete()
