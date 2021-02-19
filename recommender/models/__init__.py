# pylint: disable=missing-module-docstring
from .access_type import AccessType
from .access_mode import AccessMode
from .action import Action
from .category import Category
from .life_cycle_status import LifeCycleStatus
from .platform import Platform
from .provider import Provider
from .recommendation import Recommendation
from .root import Root
from .sars import Sars
from .scientific_domain import ScientificDomain
from .service import Service
from .source import Source
from .state import State
from .target import Target
from .target_user import TargetUser
from .trl import Trl
from .user import User
from .user_action import UserAction

MP_DUMP_MODEL_CLASSES = [
    Category,
    Provider,
    ScientificDomain,
    Platform,
    TargetUser,
    AccessMode,
    AccessType,
    Trl,
    LifeCycleStatus,
    User,
    Service,
]
