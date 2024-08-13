"""Unsupervised models for anomaly detection."""

from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .pipeline import preprocessing_pipeline


def get_model(model_name):
    """
    Get an unsupervised model by name.
    Supported models:
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


def create_pipeline(model_name):
    """
    Create a pipeline for an unsupervised model by name.
    The pipeline includes preprocessing, scaling, dimensionality reduction, and the model.
    """
    model = get_model(model_name)
    return Pipeline(
        [
            ("preprocessing", preprocessing_pipeline),
            ("scaler", StandardScaler()),
            (
                "dim_reduction",
                PCA(n_components=2),
            ),  # Reduce dimensions for visualization
            ("model", model),
        ]
    )
