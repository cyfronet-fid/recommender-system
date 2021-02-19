# pylint: disable=invalid-name, too-many-locals, no-member

"""Functions responsible for transforming MongoEngine models from database
 to Pandas dataframes
"""
import pickle
import random
import pandas as pd
from tqdm.auto import tqdm

from recommender.models import User, Service

SERVICE_COLUMNS = [
    "name",
    "description",
    "tagline",
    "countries",
    "categories",
    "providers",
    "resource_organisation",
    "scientific_domains",
    "platforms",
    "target_users",
    "access_modes",
    "access_types",
    "trls",
    "life_cycle_statuses",
]

USER_COLUMNS = ["scientific_domains", "categories"]

USERS = "users"
SERVICES = "services"
LABELS = "labels"


def service_to_df(service, save_df=False):
    """It transform MongoEngine Service object into Pandas dataframe"""

    df = pd.DataFrame(columns=SERVICE_COLUMNS)

    row_dict = {
        "name": service.name,
        "description": service.description,
        "tagline": service.tagline.split(", "),
        "countries": list(service.countries),
        "categories": [category.name for category in service.categories],
        "providers": [provider.name for provider in service.providers],
        "resource_organisation": service.resource_organisation.name,
        "scientific_domains": [
            scientific_domain.name for scientific_domain in service.scientific_domains
        ],
        "platforms": [platform.name for platform in service.platforms],
        "target_users": [
            (target_user.name, target_user.description)
            for target_user in service.target_users
        ],
        "access_modes": [
            (access_mode.name, access_mode.description)
            for access_mode in service.access_modes
        ],
        "access_types": [
            (access_type.name, access_type.description)
            for access_type in service.access_types
        ],
        "trls": [(trl.name, trl.description) for trl in service.trls],
        "life_cycle_statuses": [
            (life_cycle_status.name, life_cycle_status.description)
            for life_cycle_status in service.life_cycle_statuses
        ],
    }

    df = df.append(row_dict, ignore_index=True)
    if save_df:
        service.dataframe = pickle.dumps(df)
        service.save()

    return df


def user_to_df(user, save_df=False):
    """It transform MongoEngine User object into Pandas dataframe"""

    df = pd.DataFrame(columns=USER_COLUMNS)

    row_dict = {
        "categories": [category.name for category in user.categories],
        "scientific_domains": [
            scientific_domain.name for scientific_domain in user.scientific_domains
        ],
    }

    df = df.append(row_dict, ignore_index=True)
    if save_df:
        user.dataframe = pickle.dumps(df)
        user.save()

    return df


def calculate_dfs_for_users_and_services():
    """Calculate dataframes for all users and services"""
    for user in tqdm(User.objects):
        user_to_df(user, save_df=True)
    for service in tqdm(Service.objects):
        service_to_df(service, save_df=True)


def create_raw_dataset(save_df=False):
    """Creates balanced dataset that consist of pairs user-service.

    If there is n users and each of them has on average k services
    then the final dataset will consist of 2kn examples
    (not just kn because for each k positive examples of services
    oredered by a user there are generated also k negative services
    not ordered by a user).

    Time and space complexity of this algorithm is O(kn)
    """

    users_df_rows = []
    services_df_rows = []
    labels_df_rows = []

    ordered_class_df = pd.DataFrame({"ordered": [True]})
    not_ordered_class_df = pd.DataFrame({"ordered": [False]})

    for user in tqdm(User.objects):
        user_df = user_to_df(user, save_df=save_df)

        # Positive examples
        ordered_services = user.accessed_services

        for service in ordered_services:
            service_df = service_to_df(service, save_df=save_df)
            users_df_rows.append(user_df)
            services_df_rows.append(service_df)
            labels_df_rows.append(ordered_class_df)

        # Negative examples (same amount as positive - classes balance)
        ordered_services_ids = [s.id for s in ordered_services]
        all_not_ordered_services = list(Service.objects(id__nin=ordered_services_ids))
        k = min(len(ordered_services), len(all_not_ordered_services))
        not_ordered_services = random.sample(all_not_ordered_services, k=k)

        for service in not_ordered_services:
            service_df = service_to_df(service, save_df=save_df)
            users_df_rows.append(user_df)
            services_df_rows.append(service_df)
            labels_df_rows.append(not_ordered_class_df)

    users_df = pd.concat(users_df_rows, ignore_index=True)
    services_df = pd.concat(services_df_rows, ignore_index=True)
    labels_df = pd.concat(labels_df_rows, ignore_index=True)

    raw_dataset = {USERS: users_df, SERVICES: services_df, LABELS: labels_df}

    return raw_dataset
