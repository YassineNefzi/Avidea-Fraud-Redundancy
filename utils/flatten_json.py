"""Module to flatten a JSON column in a DataFrame."""

import pandas as pd
import json


def flatten_json_column(df, column_name):
    """Flatten a JSON column in a DataFrame."""

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
