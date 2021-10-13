# pylint: disable=missing-module-docstring, missing-class-docstring
# pylint: disable=missing-function-docstring

from mongoengine import StringField, Document, DateField, ListField

from recommender.models.step_metadata import StepMetadata


class PipelineMetadata(Document):
    type = StringField()
    start_time = DateField()
    end_time = DateField()
    steps = ListField(StepMetadata)

    def status(self):
        pass
