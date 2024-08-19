import pandas as pd
from src.utils.preprocessor import process_file
from src.utils.flatten_df import flatten_dataframe


def test_flatten_dataframe():
    df = pd.read_csv("C:\\Users\\ynyas\\AI\\Avidea\\data\\report5.csv")
    df_flattened = flatten_dataframe(df)

    relevant_columns = [
        "region",
        "vehicles_plate_number",
        "casualties_identity",
        "casualties_health_institution",
    ]
    group_column = "report_number"
    date_column = "report_accident_date"

    output_file = process_file(
        df_flattened, relevant_columns, group_column, date_column
    )

    print(f"Redundancy analysis output saved to: {output_file}")
