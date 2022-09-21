# pylint: disable=too-few-public-methods

"""Base Explanation Interface"""

from abc import ABC


class Explanation(ABC):
    """This class establishes the interface for explanations of recommendations"""

    def __init__(self, long, short):
        self.long = long
        self.short = short
