# pylint: disable=missing-module-docstring, missing-class-docstring

from mongoengine import EmbeddedDocumentField

from .root import Root
from .target import Target


class Source(Target):
    root = EmbeddedDocumentField(Root, blank=True)
