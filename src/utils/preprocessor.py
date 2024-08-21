"""Preprocesses the data for the redundancy checker module."""

import pandas as pd

from ..modules.redundancy_checker import RedundancyChecker
from ..config.constants import OUTPUT_FILE
from .preprocessing_functions import (
    to_dataframe,
    to_numerical,
    filter_semi_duplicated_rows,
)
from ..models.pipeline import preprocessing_pipeline


def process_file(
    df_flattened: pd.DataFrame,
    relevant_columns: str,
    group_column: str,
    date_column: str = "report_accident_date",
):
    """Process the flattened DataFrame for redundancy analysis."""

    if group_column not in df_flattened.columns:
        raise KeyError(f"Column not found: {group_column}")

    df_filtered = df_flattened[relevant_columns + [group_column, date_column]]

    checker = RedundancyChecker(df_filtered, relevant_columns, group_column)
    checker.preprocess()

    output_file = OUTPUT_FILE
    checker.save_results(output_file)

    return output_file


def preprocessor(df: pd.DataFrame) -> pd.DataFrame:
    """Function to pre-process the data.
    Args:
    df (pd.DataFrame): Input DataFrame.
    Returns:
    pd.DataFrame: DataFrame containing the pre-processed data.
    """
    df_transformed = preprocessing_pipeline.fit_transform(df)
    df_transformed = to_dataframe(df_transformed)
    df_transformed = to_numerical(df_transformed)

    return df_transformed
