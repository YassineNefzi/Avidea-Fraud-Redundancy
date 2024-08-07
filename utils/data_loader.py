import pandas as pd
from flatten_json import flatten_json_column


class DataLoader:
    def __init__(self, input_data):
        self.input_data = input_data

    def load_and_flatten(self):
        data = self.input_data
        if 'vehicles' in data.columns:
            data = flatten_json_column(data, 'vehicles')
        if 'casualties' in data.columns:
            data = flatten_json_column(data, 'casualties')
        return data
