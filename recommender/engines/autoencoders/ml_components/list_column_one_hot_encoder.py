# pylint: disable=keyword-arg-before-vararg, invalid-name, unused-argument, missing-function-docstring

"""This module contains custom encoder needed for preprocessing"""

from sklearn.base import BaseEstimator, TransformerMixin


class ListColumnOneHotEncoder(BaseEstimator, TransformerMixin):
    """This encoder get column(s) of rows filled with list of values
    and transform them using on-hot encoding with new column for each
    unique value in all lists of the column"""

    def __init__(self):
        self.unique_values = {}
        self.old_columns = None

    def fit(self, X, y=None, *args, **kwargs):
        for column_name in X.columns:
            self.unique_values[column_name] = set()

        for column_name in X.columns:
            for row in X[column_name]:
                for entry in row:
                    self.unique_values.get(column_name).add(entry)

            sorted_list = sorted(list(self.unique_values.get(column_name)))
            self.unique_values[column_name] = sorted_list

        return self

    def transform(self, X, *args, **kwargs):
        self.old_columns = X.columns
        for old_column in self.old_columns:
            for unique_value in self.unique_values[old_column]:
                new_column = (old_column, unique_value)
                X.insert(len(X.columns), new_column, 0)

        X = X.apply(self.set_values, axis="columns")

        X.drop(columns=self.old_columns, inplace=True)
        return X

    def set_values(self, row):
        """utility method used for transforming each row"""
        for old_column in self.old_columns:
            for unique_value in row[old_column]:
                new_column = (old_column, unique_value)
                if row.get(new_column) is not None:
                    row[new_column] = 1
        return row
