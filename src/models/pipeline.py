"""This module contains the preprocessing pipeline for the unsuperivsed model."""

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from ..utils.preprocessing_classes import (
    DropColumns,
    DebugStep,
    ReplaceMissingWithNaN,
    ColumnThresholdDropper,
    IdentityConverter,
    PoliceGuardTransformer,
    DateSplitter,
    CustomOrdinalEncoder,
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
        # ("plate_number_fe", PlateNumberFeatureEngineer()),
        # ("debug_step6", DebugStep()),
        ("date_splitter", DateSplitter(date_columns=DATE_COLS)),
        ("debug_step7", DebugStep()),
        (
            "imputer",
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
        ("ordinal_encoder", CustomOrdinalEncoder(columns=CAT_COLS_AFTER_IMPUTING)),
        ("scaler", StandardScaler()),
    ]
)
