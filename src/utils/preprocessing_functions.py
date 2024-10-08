"""This module contains utility functions for the pre-processing process."""

import pandas as pd
import json

from ..config.constants import (
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    PLATE_NUMBER_FEATURES,
    NUMERICAL_COLS_AFTER_PIPELINE,
)


def flatten_json_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Function to flatten a JSON column in a DataFrame.
    Args:
    df (pd.DataFrame): Input DataFrame.
    column_name (str): Name of the JSON column to flatten.
    Returns:
    pd.DataFrame: DataFrame with the JSON column flattened.
    """

    df[column_name] = df[column_name].apply(json.loads)

    df_exploded = df.explode(column_name)

    df_normalized = pd.json_normalize(df_exploded[column_name])

    df_normalized.columns = [f"{column_name}_{col}" for col in df_normalized.columns]

    df_merged = pd.concat(
        [df_exploded.reset_index(drop=True), df_normalized.reset_index(drop=True)],
        axis=1,
    )

    df_merged.drop(column_name, axis=1, inplace=True)

    return df_merged


def filter_semi_duplicated_rows(
    df: pd.DataFrame, report_col: str, vehicle_col: str, casualty_col: str
) -> pd.DataFrame:
    """Filter out semi-duplicated rows that represent valid accidents with multiple vehicles or casualties.
    Args:
    df (pd.DataFrame): Input flattened DataFrame.
    report_col (str): Column name representing the report number.
    vehicle_col (str): Column name representing the vehicle ID.
    casualty_col (str): Column name representing the casualty ID.
    Returns:
    pd.DataFrame: Filtered DataFrame with semi-duplicated rows removed.
    """

    grouped_df = (
        df.groupby(report_col)
        .agg(
            num_vehicles=(vehicle_col, "nunique"),
            num_casualties=(casualty_col, "nunique"),
        )
        .reset_index()
    )

    valid_reports = grouped_df[
        (grouped_df["num_vehicles"] <= 4) & (grouped_df["num_casualties"] <= 4)
    ]

    filtered_df = df[df[report_col].isin(valid_reports[report_col])]

    return filtered_df


def to_dataframe(df) -> pd.DataFrame:
    """Function to transform the pre-processed data into a DataFrame.
    Returns:
    pd.DataFrame: DataFrame containing the pre-processed data.
    """

    num_features = [f"num__{col}" for col in NUMERICAL_COLS]
    cat_features = [f"cat__{col}" for col in CATEGORICAL_COLS]
    # plate_features = [col for col in PLATE_NUMBER_FEATURES]

    remainder_features = [
        "remainder__casualties_identity",
        "remainder__is_police",
        "remainder__report_accident_date_year",
        "remainder__report_accident_date_month",
        "remainder__report_accident_date_day",
        "remainder__report_date_year",
        "remainder__report_date_month",
        "remainder__report_date_day",
    ]

    all_features = num_features + cat_features + remainder_features

    df_transformed = pd.DataFrame(df, columns=all_features)

    return df_transformed


def to_numerical(df: pd.DataFrame) -> pd.DataFrame:
    """Function to convert columns to numerical after the pipeline.
    Args:
    df (pd.DataFrame): Input DataFrame.
    Returns:
    pd.DataFrame: DataFrame with columns converted to numerical.
    """

    df[NUMERICAL_COLS_AFTER_PIPELINE] = df[NUMERICAL_COLS_AFTER_PIPELINE].apply(
        pd.to_numeric
    )
    return df
