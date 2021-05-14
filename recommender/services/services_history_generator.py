# pylint: disable=invalid-name, no-member

"""Services History Generator implementation"""

import copy
from typing import List, Any

from recommender.models import UserAction, User, Service


def _bfs_ordered_finder(uas: List[UserAction]) -> bool:
    """
    Check if there exist ordering user action in provided uas.
    If not, check these user actions' children.

    Args:
        uas: List of UserAction objects.

    Returns:
        order: ordered service or None if this action doesn't lead to service
         ordering.
    """

    # Recursion stop condition
    if not uas:
        return False

    # Try to find ordered service
    for ua in uas:
        if ua.action.order:
            return True

            # Get children of all provided user actions
    new_uas = []
    for ua in uas:
        target_id = ua.target.visit_id
        children = list(UserAction.objects(source__visit_id=target_id))
        new_uas += children
    new_uas = copy.deepcopy(new_uas)

    # Try to find ordered service in children
    order = _bfs_ordered_finder(new_uas)

    return order


def leads_to_order(user_action: UserAction) -> bool:
    """
    Check if this user action leads to the service order.

    Args:
        user_action: UserAction object.

    Returns:
        ordered: Order flag.
    """

    ordered = _bfs_ordered_finder([user_action])

    return ordered


def _get_ruas_services(ruas):
    """
    Get services from root user actions.

    Args:
        ruas: Root user actions.

    return:

    """

    services = [ua.source.root.service for ua in ruas]

    return services


def _list_difference(minuend: List[Any], subtrahend: List[Any]) -> List[Any]:
    """
    Get difference between two lists preserving order of items.

    Args:
        minuend: the minuend list.
        subtrahend: the subtrahend list.

    Returns:
        difference: Lists difference.
    """

    subtrahend = set(subtrahend)
    difference = [item for item in minuend if item not in subtrahend]

    return difference


def concat_histories(accessed_services, root_uas):
    """
    This function get user's accessed services (from the DB dump) and combine
     it with clicked services that correspond to root user actions.

    Essentially it appends accessed services on the beginning of the list of
     services from root user actions but it filter ordered services
     duplicates out of it.

    Args:
        accessed_services: User's accessed services (from DB dump)
        root_uas: root user actions that correspond to clicked services

    Return:

    """

    # Get clicked services
    clicked_services = _get_ruas_services(root_uas)

    # Get ordered services
    root_uas_leading_to_order = list(filter(leads_to_order, root_uas))
    ordered_services = _get_ruas_services(root_uas_leading_to_order)

    # Smart concat
    ld = _list_difference(accessed_services, ordered_services)
    services_history = ld + clicked_services

    return services_history


def generate_services_history(user: User) -> List[Service]:
    """
    Create clicked services history for the given user.
    It consist of:
        -> user.accessed_services: where are all services ordered by
           the user in MP before sending the DB dump from MP to Recommender.
        -> clicked_services from root user actions - that has been received
           by recommender from MP during normal operation. They consist of
           two subtypes:
            -> ordered services
            -> not ordered services (just clicked)

    This function get all clicked services in the temporal order and
    add on the beginning of it all ordered services that exist only
    in the MP DB dump (user.accessed_services) - but only if they
    do not repeat in the clicked services as ordered.

    As a result the temporal order of services from these
     two sources (dump, user actions) is preserved.

    Args:
        user: MongoEngine User object.

    Returns:
        services_history: history of user's clicked/ordered services.
    """

    #  Get accessed services
    accessed_services = user.accessed_services

    # Get clicked services (ordered or not)
    root_uas = list(
        UserAction.objects(
            source__root__type__="recommendation_panel", user=user
        ).order_by("+timestamp")
    )

    services_history = concat_histories(accessed_services, root_uas)

    return services_history
