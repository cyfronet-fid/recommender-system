# pylint: disable=keyword-arg-before-vararg, invalid-name, unused-argument, missing-function-docstring, line-too-long

"""This module contains custom encoder needed for preprocessing"""

from __future__ import annotations
from warnings import simplefilter
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

# There is a performance warning because pandas insert is used, but it doesn't matter at that scale.
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class ListColumnOneHotEncoder(BaseEstimator, TransformerMixin):
    """This encoder gets column(s) of rows filled with list of values
    and transform them using on-hot encoding with new column for each
    unique value in all lists of the column"""

    def __init__(self):
        self.unique_values = {}
        self.old_columns = None

    def fit(self, df: pd.DataFrame) -> ListColumnOneHotEncoder:
        for column_name in df.columns:
            self.unique_values[column_name] = set()

        for column_name in df.columns:
            for row in df[column_name]:
                for entry in row:
                    self.unique_values.get(column_name).add(entry)

            sorted_list = sorted(list(self.unique_values.get(column_name)))
            self.unique_values[column_name] = sorted_list
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.old_columns = df.columns
        for old_column in self.old_columns:
            for unique_value in self.unique_values[old_column]:
                new_column = (old_column, unique_value)
                df.insert(len(df.columns), new_column, 0)

        df = df.apply(self.set_values, axis="columns")

        df.drop(columns=self.old_columns, inplace=True)
        return df

    def set_values(self, row: pd.Series) -> pd.Series:
        """utility method used for transforming each row"""
        for old_column in self.old_columns:
            for unique_value in row[old_column]:
                new_column = (old_column, unique_value)
                if row.get(new_column) is not None:
                    row[new_column] = 1
        return row
