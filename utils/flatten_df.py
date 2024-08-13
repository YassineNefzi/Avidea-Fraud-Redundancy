from .data_loader import DataLoader


def flatten_dataframe(df):
    """Flatten the JSON columns in the DataFrame."""

    if "report_id" not in df.columns:
        raise KeyError("Column not found: report_id")

    data_loader = DataLoader(df)
    return data_loader.load_and_flatten()
