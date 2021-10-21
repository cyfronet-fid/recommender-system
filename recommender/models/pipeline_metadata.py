# pylint: disable=missing-module-docstring, missing-class-docstring
# pylint: disable=missing-function-docstring

from mongoengine import (
    StringField,
    Document,
    ListField,
    DateTimeField,
    EmbeddedDocumentField,
)

from recommender.models.step_metadata import StepMetadata, Status


class PipelineMetadata(Document):
    type = StringField()
    start_time = DateTimeField()
    end_time = DateTimeField()
    steps = ListField(EmbeddedDocumentField(StepMetadata))

    def status(self):
        if all(step.status == Status.COMPLETED for step in self.steps):
            return Status.COMPLETED
        return Status.NOT_COMPLETED
