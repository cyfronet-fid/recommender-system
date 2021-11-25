# pylint: disable=invalid-name, missing-module-docstring

from recommender.migrate.base_migration import BaseMigration


class RemoveUnusedCollections(BaseMigration):
    """
    Drop ScikitLearnTransformer, PytorchDataset and Pytorch module
    collection as they are no longer used (replaced by MLComponents)
    """

    def up(self):
        self.pymongo_db.scikit_learn_transformer.drop()
        self.pymongo_db.pytorch_dataset.drop()
        self.pymongo_db.pytorch_module.drop()

    def down(self):
        # You cannot create a collection in mongodb unless you insert a document into it
        pass
