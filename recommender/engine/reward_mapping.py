"""This module contains logic needed for mapping user actions to rewards"""


def ua_to_reward_id(_user_action):
    """
    This function should map user_action to the reward id.
    Mapping should use following fields of user_action:
    user_action.source.page_id,
    user_action.target.page_id,
    user_action.action.type,
    user_action.action.text,
    user_action.action.order

    For now it just return generic reward id.
    """

    return "generic_reward_id"
