# pylint: disable=invalid-name, no-member

"""This module contains functions used for
saving specific data from the database
into json files.
This data is needed for seeding faker in factories.
It makes them more realistic.
"""

import os
import json
import inflection
from recommender.models import Service

from definitions import ROOT_DIR

FAKER_SEEDS_PATH = "tests/factories/marketplace/faker_seeds"


def save_json_file(file_name, data):
    """It is utility function used in dump functions below"""
    file_path = os.path.join(ROOT_DIR, FAKER_SEEDS_PATH, file_name)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, sort_keys=True)


def dump_names_descs(Clazz):
    """It dumps names and descriptions of database instances of given
    Mongoengine class into json file. This file is used for seeding factories"""
    data = {obj.name: obj.description for obj in Clazz.objects}

    formatted_clazz_name = inflection.underscore(Clazz.__name__)
    file_name = f"{formatted_clazz_name}_names_descs.json"
    save_json_file(file_name, data)


def dump_names(Clazz):
    """It dumps names of database instances of given
    Mongoengine class into json file. This file is used for seeding factories"""
    data = list({obj.name for obj in Clazz.objects})
    formatted_clazz_name = inflection.underscore(Clazz.__name__)
    file_name = f"{formatted_clazz_name}_names.json"
    save_json_file(file_name, data)


def dump_taglines():
    """It dumps taglines of database instances of given
    Mongoengine class into json file. This file is used for seeding factories"""
    data = list({service.tagline for service in Service.objects})
    file_name = "service_taglines.json"
    save_json_file(file_name, data)
