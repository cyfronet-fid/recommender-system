# pylint: disable=no-member

"""Functions to handle marketplace database dump"""

from inflection import underscore, pluralize

from recommender.models import MP_DUMP_MODEL_CLASSES


def load_mp_dump(data):
    """Loads database dump sent and serialized by Marketplace into mongodb"""

    for model in MP_DUMP_MODEL_CLASSES:
        json_key = pluralize(underscore(model.__name__))
        model_fields = list(model._fields.keys())
        for json_dict in data[json_key]:
            # Only include fields defined in the mongonengine model
            json_dict = {
                key: value for key, value in json_dict.items() if key in model_fields
            }
            model(**json_dict).save()


def drop_mp_dump():
    """Drops part of mongodb which consists of classes sent by Marketplace"""

    for model in MP_DUMP_MODEL_CLASSES:
        for obj in model.objects:
            obj.delete()
