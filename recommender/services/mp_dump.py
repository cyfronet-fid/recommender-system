# pylint: disable=no-member

"""Functions to handle marketplace database dump"""

from inflection import underscore, pluralize

from recommender.models import MP_DUMP_MODEL_CLASSES


def load_mp_dump(data):
    """Loads database dump sent and serialized by Marketplace into mongodb"""

    for model in MP_DUMP_MODEL_CLASSES:
        json_key = pluralize(underscore(model.__name__))
        for json_dict in data[json_key]:
            model(**json_dict).save()


def drop_mp_dump():
    """Drops part of mongodb which consits of classes sent by Marketplace"""

    for model in MP_DUMP_MODEL_CLASSES:
        for obj in model.objects:
            obj.delete()
