# pylint: disable=invalid-name

"""This module contains functions used for
loading specific data from json files.
This data is needed for seeding faker in factories.
It makes them more realistic.
"""


import os
import json
import inflection

from definitions import ROOT_DIR

FAKER_SEEDS_PATH = "tests/factories/marketplace/faker_seeds"


def load_json_file(file_name):
    """It is utility function used in load functions below"""
    file_path = os.path.join(ROOT_DIR, FAKER_SEEDS_PATH, file_name)

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_names_descs(Clazz):
    """It loads names and descriptions of database instances of given
    Mongoengine class from json file. This data is used for seeding factories"""
    formatted_clazz_name = inflection.underscore(Clazz.__name__)
    file_name = f"{formatted_clazz_name}_names_descs.json"
    return load_json_file(file_name)


def load_names(Clazz):
    """It loads names of database instances of given
    Mongoengine class from json file. This data is used for seeding factories"""
    formatted_clazz_name = inflection.underscore(Clazz.__name__)
    file_name = f"{formatted_clazz_name}_names.json"
    return load_json_file(file_name)


def load_taglines():
    """It loads tagline of database instances of Mongoengine Service
    class from json file. This data is used for seeding factories"""
    file_name = "service_taglines.json"
    return load_json_file(file_name)
