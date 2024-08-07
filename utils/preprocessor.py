"""Preprocesses the data for the redundancy checker module."""

import pandas as pd

from .data_loader import DataLoader
from modules.redundancy_checker import RedundancyChecker
from config.constants import OUTPUT_FILE


def flatten_dataframe(df):
    """Flatten the JSON columns in the DataFrame."""

    if "report_id" not in df.columns:
        raise KeyError("Column not found: report_id")

    data_loader = DataLoader(df)
    return data_loader.load_and_flatten()


def process_file(df_flattened, relevant_columns, group_column):
    """Process the flattened DataFrame for redundancy analysis."""

    if group_column not in df_flattened.columns:
        raise KeyError(f"Column not found: {group_column}")

    df_filtered = df_flattened[relevant_columns + [group_column]]

    checker = RedundancyChecker(df_filtered, relevant_columns, group_column)
    checker.preprocess()

    output_file = OUTPUT_FILE
    checker.save_results(output_file)

    return output_file
