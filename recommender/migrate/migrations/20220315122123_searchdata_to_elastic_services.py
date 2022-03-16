# pylint: disable=invalid-name, missing-module-docstring, line-too-long

from copy import deepcopy
from mongoengine import QuerySet
from tqdm import tqdm
from recommender.services.fts import filter_services
from recommender.models.search_data import SearchData
from recommender.migrate.base_migration import BaseMigration
from logger_config import get_logger

logger = get_logger(__name__)


class SearchDataToElasticServices(BaseMigration):
    """If recommendation and state documents do not have elastic_services, create it using searchdata"""

    def up(self):
        (
            recs_for_up,
            len_recs,
            states_for_up,
            len_states,
        ) = self.get_update_candidates()

        if len_recs:
            logger.info("Found %s recommendations documents to update", len_recs)

            for rec in tqdm(list(recs_for_up), desc="Recommendations to update"):
                elastic_services = self.get_elastic_services_from_sd(rec["search_data"])

                self.pymongo_db.recommendation.update_one(
                    {"_id": {"$eq": rec["_id"]}},
                    {"$set": {"elastic_services": elastic_services.distinct("id")}},
                )

        if len_states:
            logger.info("Found %s states documents to update", len_states)

            for state in tqdm(list(states_for_up), desc="States to update"):
                elastic_services = self.get_elastic_services_from_sd(
                    state["search_data"]
                )

                self.pymongo_db.state.update_one(
                    {"_id": {"$eq": state["_id"]}},
                    {"$set": {"elastic_services": elastic_services.distinct("id")}},
                )

    def down(self):
        # Information about which documents didn't have elastic_services before migration took place is lost
        pass

    def get_update_candidates(self):
        """Get candidates for update"""

        recs_for_up = self.pymongo_db.recommendation.find(
            {"elastic_services": {"$exists": False}}
        )

        states_for_up = self.pymongo_db.state.find(
            {"elastic_services": {"$exists": False}}
        )

        len_recs = len(list(deepcopy(recs_for_up)))
        len_states = len(list(deepcopy(states_for_up)))

        return recs_for_up, len_recs, states_for_up, len_states

    def get_elastic_services_from_sd(self, search_data: SearchData) -> QuerySet:
        """Get elastic_services from search_data"""
        search_data_dict = self.pymongo_db.search_data.find_one({"_id": search_data})

        sd = SearchData(
            categories=search_data_dict["categories"],
            geographical_availabilities=search_data_dict["geographical_availabilities"],
            providers=search_data_dict["providers"],
            related_platforms=search_data_dict["related_platforms"],
            scientific_domains=search_data_dict["scientific_domains"],
            target_users=search_data_dict["target_users"],
            q=search_data_dict.get(
                "q", ""
            ),  # Synthetic search_data does not have 'q' set
        )

        elastic_services = filter_services(sd)
        return elastic_services
