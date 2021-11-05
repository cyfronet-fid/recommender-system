from typing import Tuple

from graphviz import Digraph

from recommender.models import UserAction


def _get_visit_ids(user_action: UserAction) -> Tuple[str, str]:
    """
    Get source and target visit ids of the user action.

    Args:
        user_action: User action object.
    Returns:
        visit_ids: Tuple of the source and target visit ids.
    """

    rua_svid = user_action.source.visit_id
    rua_tvid = user_action.target.visit_id

    visit_ids = (str(rua_svid), str(rua_tvid))

    return visit_ids


def _generate_tree(user_action: UserAction, graph: Digraph) -> Digraph:
    """
    Utility function implementing recursive user actions tree generation.

    Args:
        user_action: User action that is the root of the tree.
        graph: Graph accumulator.
    """

    ua_svid, ua_tvid = _get_visit_ids(user_action)

    graph.node(ua_svid, label=ua_svid[:3])
    graph.node(ua_tvid, label=ua_tvid[:3])
    if user_action.action.order:
        color = "red"
        label = "Order"
    else:
        color = "black"
        label = ""

    if user_action.source.root is not None:
        service_id = user_action.source.root.service.id
        label = f"service id: {service_id}"
        color = "green"

    graph.edge(ua_svid, ua_tvid, color=color, label=label)

    children = list(UserAction.objects(source__visit_id=ua_tvid))
    for ua in children:
        _generate_tree(ua, graph)

    return graph


def generate_tree(user_action: UserAction) -> Digraph:
    """
    This method is used for generating the user actions' tree
    rooted in the user action given as a parameter.

    Args:
        user_action: User action that is the root of the tree.
    """

    graph = Digraph(comment="User Actions Tree")
    return _generate_tree(user_action, graph)