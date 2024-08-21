"""This module contains utility classes for the pre-processing process."""

import numpy as np
import pandas as pd
from hashlib import blake2b
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


class DropColumns(BaseEstimator, TransformerMixin):
    """
    Transformer to drop specified columns from the DataFrame.

    Parameters:
    columns (list): List of columns to drop.
    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns)


class ReplaceMissingWithNaN(BaseEstimator, TransformerMixin):
    """
    Transformer to replace specified missing values with np.nan.

    This can handle various representations of missing values, including None,
    empty strings, and any other specified representations.

    Parameters:
    missing_values (list): List of values to replace with np.nan. Default includes
                           None and empty strings.
    """

    def __init__(self, missing_values=None):
        # Default missing values to be replaced with np.nan
        if missing_values is None:
            self.missing_values = [None, ""]
        else:
            self.missing_values = missing_values

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Replacing None and NaN with np.nan
        for missing_value in self.missing_values:
            if missing_value is None:
                X = X.applymap(lambda x: np.nan if pd.isnull(x) else x)
            else:
                X.replace(missing_value, np.nan, inplace=True, regex=False)
        return X


class ColumnThresholdDropper(BaseEstimator, TransformerMixin):
    """
    Transformer to drop columns with missing values above a specified threshold.

    Parameters:
    threshold (float): Threshold for missing values, above which columns are dropped.
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.loc[:, X.isnull().mean() <= self.threshold]


class IdentityConverter(BaseEstimator, TransformerMixin):
    """
    Transformer to convert the 'casualties_identity' column from string to integer.
    Missing values are first imputed with a specified strategy.

    Parameters:
    impute_strategy (str): Strategy to use for imputing missing values (default is 'constant').
    fill_value (str): The value to replace missing values with if the strategy is 'constant' (default is '0').
    """

    def __init__(self, impute_strategy="constant", fill_value="0"):
        self.imputer = SimpleImputer(strategy=impute_strategy, fill_value=fill_value)

    def fit(self, X, y=None):
        if "casualties_identity" in X.columns:
            self.imputer.fit(X[["casualties_identity"]])
        return self

    def transform(self, X):
        if "casualties_identity" in X.columns:
            X[["casualties_identity"]] = self.imputer.transform(
                X[["casualties_identity"]]
            )
            X["casualties_identity"] = (
                pd.to_numeric(X["casualties_identity"], errors="coerce")
                .fillna(0)
                .astype(int)
            )
        return X


class PoliceGuardTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to handle police and national guard HQ information.
    Adds a new column 'is_police' and drops 'police_hq' and 'national_guard_hq' columns.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "police_hq" in X.columns:
            X["is_police"] = X["police_hq"].notnull().astype(int)
        columns_to_drop = [
            col for col in ["police_hq", "national_guard_hq"] if col in X.columns
        ]
        X = X.drop(columns=columns_to_drop)
        return X


class DateSplitter(BaseEstimator, TransformerMixin):
    """
    Transformer to split date columns into separate year, month, and day columns.

    Parameters:
    date_columns (list): List of date columns to split.
    """

    def __init__(self, date_columns=None):
        self.date_columns = date_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.date_columns:
            X[col] = pd.to_datetime(X[col], errors="coerce")
            X[f"{col}_year"] = X[col].dt.year
            X[f"{col}_month"] = X[col].dt.month
            X[f"{col}_day"] = X[col].dt.day
            X = X.drop(columns=[col])
        return X


class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(
        self, columns=None, method="zscore", threshold=3.0, strategy="nan", value=None
    ):
        """
        Parameters:
        - columns: List of columns to check for outliers. If None, all columns are checked.
        - method: Method to detect outliers ('zscore' or 'iqr').
        - threshold: Threshold for determining outliers (z-score value or IQR multiplier).
        - strategy: Strategy to handle outliers ('nan', 'value', or 'remove').
        - value: Value to replace outliers with if strategy is 'value'.
        """
        self.columns = columns
        self.method = method
        self.threshold = threshold
        self.strategy = strategy
        self.value = value

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.columns = X.select_dtypes(include=[np.number]).columns.tolist()
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        X = X.copy()
        cols = (
            self.columns
            if self.columns
            else X.select_dtypes(include=[np.number]).columns
        )

        for col in cols:
            if self.method == "zscore":
                z_scores = np.abs((X[col] - X[col].mean()) / X[col].std())
                mask = z_scores > self.threshold
            elif self.method == "iqr":
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                mask = (X[col] < (Q1 - self.threshold * IQR)) | (
                    X[col] > (Q3 + self.threshold * IQR)
                )

            if self.strategy == "nan":
                X.loc[mask, col] = np.nan
            elif self.strategy == "value":
                X.loc[mask, col] = self.value
            elif self.strategy == "remove":
                X = X[~mask]
                X = X.reset_index(drop=True)

        return X


class PlateNumberFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer to engineer features from the plate number column.

    Features created:
    - plate_length: Length of the plate number.
    - num_digits: Number of numeric characters.
    - num_alpha: Number of alphabetic characters.
    - has_hyphen: Whether the plate contains a hyphen.
    - starts_with_alpha: Whether the plate starts with an alphabetic character.
    - alpha_numeric_segments: Number of separate alphanumeric segments.
    - prefix: First few characters (default=3) of the plate number.
    - suffix: Last few characters (default=3) of the plate number.
    - plate_number_hash: Hash of the plate number for uniqueness.
    - digit_segments: Numeric segments of the plate number.
    - alpha_segments: Alphabetic segments of the plate number.
    - detailed_prefix: More detailed prefix (default=4).
    - detailed_suffix: More detailed suffix (default=4).
    - numeric_pattern: Exact numeric pattern sequence.
    - alpha_pattern: Exact alphabetic pattern sequence.
    """

    def __init__(
        self,
        prefix_length=3,
        suffix_length=3,
        detailed_prefix_length=4,
        detailed_suffix_length=4,
    ):
        self.prefix_length = prefix_length
        self.suffix_length = suffix_length
        self.detailed_prefix_length = detailed_prefix_length
        self.detailed_suffix_length = detailed_suffix_length

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "vehicles_plate_number" not in X.columns:
            return X

        X["vehicles_plate_number"] = X["vehicles_plate_number"].fillna("").astype(str)

        X["plate_length"] = X["vehicles_plate_number"].apply(len)
        X["plate_num_digits"] = X["vehicles_plate_number"].apply(
            lambda x: sum(c.isdigit() for c in x)
        )
        X["plate_num_alpha"] = X["vehicles_plate_number"].apply(
            lambda x: sum(c.isalpha() for c in x)
        )
        X["plate_has_hyphen"] = (
            X["vehicles_plate_number"].apply(lambda x: "-" in x).astype(int)
        )
        X["plate_starts_with_alpha"] = (
            X["vehicles_plate_number"]
            .apply(lambda x: x[0].isalpha() if x else False)
            .astype(int)
        )
        X["plate_alpha_numeric_segments"] = X["vehicles_plate_number"].apply(
            lambda x: (
                len(
                    [
                        s
                        for s in pd.Series(list(x)).groupby(
                            pd.Series(list(x)).map(str.isdigit)
                        )
                    ]
                )
                if x
                else 0
            )
        )
        X["plate_prefix"] = X["vehicles_plate_number"].apply(
            lambda x: (
                x[: self.prefix_length] if len(x) >= self.prefix_length else "No Prefix"
            )
        )
        X["plate_suffix"] = X["vehicles_plate_number"].apply(
            lambda x: (
                x[-self.suffix_length :]
                if len(x) >= self.suffix_length
                else "No Suffix"
            )
        )
        X["plate_number_hash"] = X["vehicles_plate_number"].apply(
            lambda x: blake2b(x.encode(), digest_size=16).hexdigest()
        )

        X["plate_digit_segments"] = X["vehicles_plate_number"].apply(
            lambda x: "".join(c if c.isdigit() else " " for c in x).split()
            or ["No Digits"]
        )
        X["plate_alpha_segments"] = X["vehicles_plate_number"].apply(
            lambda x: "".join(c if c.isalpha() else " " for c in x).split()
            or ["No Alpha"]
        )

        X["plate_detailed_prefix"] = X["vehicles_plate_number"].apply(
            lambda x: (
                x[: self.detailed_prefix_length]
                if len(x) >= self.detailed_prefix_length
                else "No Prefix"
            )
        )
        X["plate_detailed_suffix"] = X["vehicles_plate_number"].apply(
            lambda x: (
                x[-self.detailed_suffix_length :]
                if len(x) >= self.detailed_suffix_length
                else "No Suffix"
            )
        )
        X["plate_numeric_pattern"] = X["vehicles_plate_number"].apply(
            lambda x: "".join(c if c.isdigit() else "" for c in x) or "No Digits"
        )
        X["plate_alpha_pattern"] = X["vehicles_plate_number"].apply(
            lambda x: "".join(c if c.isalpha() else "" for c in x) or "No Alpha"
        )

        return X


class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            if self.columns:
                X[self.columns] = X[self.columns].astype(str)
                self.encoder.fit(X[self.columns])
            else:
                X = X.astype(str)
                self.encoder.fit(X)
        else:
            X = X.astype(str)
            self.encoder.fit(X)
        return self

    def transform(self, X):
        X_copy = X.copy()
        if isinstance(X_copy, pd.DataFrame):
            if self.columns:
                X_copy[self.columns] = X_copy[self.columns].astype(str)
                X_copy[self.columns] = self.encoder.transform(X_copy[self.columns])
            else:
                X_copy = X_copy.astype(str)
                X_copy = self.encoder.transform(X_copy)
        else:
            X_copy = X_copy.astype(str)
            X_copy = self.encoder.transform(X_copy)
        return X_copy


class ConvertToDataframe(BaseEstimator, TransformerMixin):
    def __init__(self, column_names=None):
        self.column_names = column_names

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.column_names = X.columns.tolist()
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            if self.column_names:
                return pd.DataFrame(X, columns=self.column_names)
            else:
                return pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            return X
        else:
            raise ValueError("Input should be either a NumPy array or a DataFrame")


class DebugStep(BaseEstimator, TransformerMixin):
    """Transformer to print the DataFrame at a specific step in the pipeline."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print("Debugging DataFrame at this step: \n")
        print(X.head())
        return X
