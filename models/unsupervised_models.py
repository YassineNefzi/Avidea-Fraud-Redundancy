"""Unsupervised models for anomaly detection."""

from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pandas as pd

from .pipeline import preprocessing_pipeline


def get_model(model_name: str):
    """
    Get an unsupervised model by name.
    Supported models and their default parameters:
    - KNN: n_neighbors=5
    - DBSCAN: eps=0.5, min_samples=5
    - One-Class SVM: kernel="rbf", gamma=0.1, nu=0.05
    - Isolation Forest: contamination=0.05, random_state=42
    """

    if model_name == "KNN":
        return NearestNeighbors(n_neighbors=5)
    elif model_name == "DBSCAN":
        return DBSCAN(eps=0.5, min_samples=5)
    elif model_name == "One-Class SVM":
        return OneClassSVM(kernel="rbf", gamma=0.1, nu=0.05)
    elif model_name == "Isolation Forest":
        return IsolationForest(contamination=0.05, random_state=42)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def train_model(model, df: pd.Dataframe):
    """
    Train the unsupervised model.
    For KNN, return the neighbors as predictions. For clustering models, return cluster labels.
    """
    if hasattr(model, "fit_predict"):
        predictions = model.fit_predict(df)
    else:
        model.fit(df)
        if hasattr(model, "kneighbors"):
            predictions = model.kneighbors(df)[1]
        else:
            raise ValueError("Unsupported model type for predictions.")

    return predictions
