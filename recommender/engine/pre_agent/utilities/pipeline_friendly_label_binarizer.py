# pylint: disable=invalid-name, unused-argument

"""This module contains custom implementation of scikit-learn's LabelBinarizer"""

from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelBinarizer


class PipelineFriendlyLabelBinarizer(TransformerMixin):
    """It turns out that original LabelBinarizer can't properly
    work inside transformer pipelines - used in preprocessing
     - due to mismatched method's signature so this custom
    class solves this problem"""

    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)

    def fit(self, x, y=0):
        """Method used for fitting transformer to the data"""
        self.encoder.fit(x)
        return self

    def transform(self, x, y=0):
        """Method used for transforming the data"""
        return self.encoder.transform(x)
