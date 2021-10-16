# pylint: disable=missing-module-docstring, missing-class-docstring
# pylint: disable=missing-function-docstring

from mongoengine import (
    StringField,
    Document,
    ListField,
    DateTimeField,
    EmbeddedDocumentField,
)

from recommender.models.step_metadata import StepMetadata


class PipelineMetadata(Document):
    type = StringField()
    start_time = DateTimeField()
    end_time = DateTimeField()
    steps = ListField(EmbeddedDocumentField(StepMetadata))

    def status(self):
        pass
