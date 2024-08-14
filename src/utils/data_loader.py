"""Module for loading and flattening data."""

import pandas as pd
from .preprocessing_functions import flatten_json_column


class DataLoader:
    """Class to load and flatten data."""

    def __init__(self, input_data):
        self.input_data = input_data

    def load_and_flatten(self) -> pd.DataFrame:
        data = self.input_data
        if "vehicles" in data.columns:
            data = flatten_json_column(data, "vehicles")
        if "casualties" in data.columns:
            data = flatten_json_column(data, "casualties")
        return data
