from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import streamlit as st


def apply_pca(df):
    """
    Apply PCA to the dataframe.
    Returns the PCA results and the explained variance ratio.
    """
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df)
    return df_pca, pca.explained_variance_ratio_


def plot_pca(pca_result, explained_variance, clusters) -> None:
    """
    Plot PCA results.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap="viridis")
    plt.title("PCA of Model Predictions")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Cluster")
    st.pyplot(plt)


def investigate_pca():
    pass
