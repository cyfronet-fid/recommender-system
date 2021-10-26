# pylint: disable-all
# TODO: Refactor to 'update' task test
#
# from inflection import underscore, pluralize
#
# from recommender.models import MP_DUMP_MODEL_CLASSES, UserAction
# from recommender.tasks import handle_db_dump
# from tests.factories.marketplace import UserFactory
# from tests.factories.user_action import UserActionFactory
# from tests.helpers import mongo_model_to_json
#
# FIELDS_NOT_INCLUDED_IN_MP = ("dataframe", "one_hot_tensor", "dense_tensor", "synthetic")
#
#
# def test_handle_db_dump(mongo, mp_dump_data):
#     UserFactory.create_batch(5)  # Creates all of the MP dump components in the DB
#     user_action = UserActionFactory()
#
#     handle_db_dump(mp_dump_data)
#
#     mongo_objects = {
#         underscore(pluralize(model.__name__)): list(model.objects)
#         for model in MP_DUMP_MODEL_CLASSES
#     }
#
#     raw_mongo_objects = {
#         k: [
#             {
#                 field: value
#                 for field, value in mongo_model_to_json(x).items()
#                 if field not in FIELDS_NOT_INCLUDED_IN_MP
#             }
#             for x in v
#         ]
#         for k, v in mongo_objects.items()
#     }
#
#     assert raw_mongo_objects == mp_dump_data
#     assert UserAction.objects.first().target == user_action.target
