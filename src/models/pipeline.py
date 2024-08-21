"""This module contains the preprocessing pipeline for the unsuperivsed model."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from ..utils.preprocessing_classes import (
    DropColumns,
    DebugStep,
    ReplaceMissingWithNaN,
    ColumnThresholdDropper,
    IdentityConverter,
    PoliceGuardTransformer,
    OutlierHandler,
    DateSplitter,
    CustomOrdinalEncoder,
    ConvertToDataframe,
)

from ..config.constants import (
    IRRELVANT_COLS,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    DATE_COLS,
    CAT_COLS_AFTER_IMPUTING,
)


preprocessing_pipeline = Pipeline(
    [
        ("drop_irrelevant", DropColumns(columns=IRRELVANT_COLS)),
        ("debug_step1", DebugStep()),
        ("replace_missing", ReplaceMissingWithNaN()),
        ("debug_step2", DebugStep()),
        ("drop_high_missing", ColumnThresholdDropper(threshold=0.5)),
        ("debug_step3", DebugStep()),
        ("identity_converter", IdentityConverter()),
        ("debug_step4", DebugStep()),
        ("police_guard_transformer", PoliceGuardTransformer()),
        ("debug_step5", DebugStep()),
        ("date_splitter", DateSplitter(date_columns=DATE_COLS)),
        ("debug_step6", DebugStep()),
        (
            "initial_imputer",
            ColumnTransformer(
                transformers=[
                    ("num", SimpleImputer(strategy="mean"), NUMERICAL_COLS),
                    (
                        "cat",
                        SimpleImputer(strategy="constant", fill_value="Not Specified"),
                        CATEGORICAL_COLS,
                    ),
                ],
                remainder="passthrough",
            ),
        ),
        # (
        #     "outlier_handler",
        #     OutlierHandler(method="zscore", threshold=3.0, strategy="nan"),
        # ),
        ("convert_to_dataframe", ConvertToDataframe()),
        # (
        #     "second_imputer",
        #     ColumnTransformer(
        #         transformers=[
        #             (
        #                 "num",
        #                 SimpleImputer(strategy="mean"),
        #                 make_column_selector(dtype_include=np.number),
        #             ),
        #         ],
        #         remainder="passthrough",
        #     ),
        # ),
        # ("convert_to_dataframe2", ConvertToDataframe()),
        (
            "ordinal_encoder",
            ColumnTransformer(
                transformers=[
                    (
                        "cat",
                        CustomOrdinalEncoder(),
                        make_column_selector(dtype_include=object),
                    ),
                ],
                remainder="passthrough",
            ),
        ),
        ("scaler", StandardScaler()),
    ]
)
