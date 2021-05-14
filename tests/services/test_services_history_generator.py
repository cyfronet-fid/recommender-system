# pylint: disable=missing-module-docstring, missing-function-docstring,
# pylint: disable=too-many-locals, invalid-name, unused-argument

from recommender.services.services_history_generator import (
    leads_to_order,
    generate_services_history,
)
from tests.factories.marketplace import UserFactory, ServiceFactory
from tests.factories.user_action import UserActionFactory


def test_retrieve_services(mongo):
    user = UserFactory(accessed_services=ServiceFactory.create_batch(5))

    accessed_services = user.accessed_services

    # Create 5 root user actions where first 3 of them
    # point to the accessed_services of indices (in the services list): 1,3,4.

    uas = UserActionFactory.create_batch(
        3, user=user, recommendation_root=True, order=False
    )
    for ua, s_i in zip(uas, [1, 3, 4]):
        ua.source.root.service = accessed_services[s_i]
        ua.save()

    uas += UserActionFactory.create_batch(
        2, user=user, recommendation_root=True, order=False
    )

    # Add tree of children user actions to the fourth user action. One of the
    # children has have order=True.
    root_ua = uas[3]

    v_id = root_ua.target.visit_id
    ua1 = UserActionFactory(user=user, source__visit_id=v_id)
    ua2 = UserActionFactory(user=user, source__visit_id=v_id)

    ua3 = UserActionFactory(user=user, source__visit_id=ua2.target.visit_id)

    ua4 = UserActionFactory(user=user, source__visit_id=ua3.target.visit_id)

    ua5 = UserActionFactory(user=user, source__visit_id=ua4.target.visit_id)
    ua6 = UserActionFactory(user=user, source__visit_id=ua4.target.visit_id)

    ua7 = UserActionFactory(user=user, source__visit_id=ua5.target.visit_id,
                            order=True)

    ua8 = UserActionFactory(user=user, source__visit_id=ua1.target.visit_id)
    ua9 = UserActionFactory(user=user, source__visit_id=ua1.target.visit_id)
    ua10 = UserActionFactory(user=user, source__visit_id=ua1.target.visit_id)
    ua11 = UserActionFactory(user=user, source__visit_id=ua1.target.visit_id)

    # Root user action's service will be added to the accessed_services
    # to simulate a new MP DB dump created after ordering this service
    # while the information about this ordering is also present in
    # user action (not only in dump)

    user.accessed_services.append(root_ua.source.root.service)
    user.save()

    non_leading = [ua1, ua6, ua8, ua9, ua10, ua11]
    assert any(leads_to_order(ua) for ua in non_leading) is False

    leading = [root_ua, ua2, ua3, ua4, ua5, ua7]
    assert all(leads_to_order(ua) for ua in leading) is True

    services_history = generate_services_history(user)
    valid_services_history = user.accessed_services[:-1] + [
        ua.source.root.service for ua in uas
    ]
    assert services_history == valid_services_history
