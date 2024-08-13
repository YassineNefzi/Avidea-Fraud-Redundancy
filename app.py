"""This module contains the Streamlit app for the Fraud Detection App."""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO

from utils.preprocessor import process_file
from utils.flatten_df import flatten_dataframe
from utils.preprocessor import preprocessor
from utils.preprocessing_functions import apply_pca, plot_pca
from models.unsupervised_models import train_model, get_model


def main():
    """Main function of the Streamlit app."""
    st.title("Fraud Detection App")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Step 1: Flatten JSON columns
        df_flattened = flatten_dataframe(df)
        st.write("Flattened DataFrame:")
        st.write(df_flattened.head())

        # Step 2: Select columns for redundancy analysis
        relevant_columns = st.multiselect(
            "Select Columns for Combination Analysis",
            options=df_flattened.columns.tolist(),
        )

        group_column = st.selectbox(
            "Select Column to Group By",
            options=df_flattened.columns.tolist(),
            index=(
                df_flattened.columns.tolist().index("report_id")
                if "report_id" in df_flattened.columns
                else 0
            ),
        )

        if st.button("Process Redundancies"):
            if not relevant_columns:
                st.error("Please select at least one column.")
            else:
                output_file = process_file(df_flattened, relevant_columns, group_column)
                st.success(
                    f"Redundancy analysis completed. Download the result file below."
                )

                with open(output_file, "rb") as file:
                    st.download_button(
                        label="Download Redundancies CSV",
                        data=file,
                        file_name="combined_redundancies_across_reports.csv",
                        mime="text/csv",
                    )

        st.write("## Model Training")
        model_name = st.selectbox(
            "Select an Unsupervised Model",
            options=["KNN", "DBSCAN", "One-Class SVM", "Isolation Forest"],
        )

        if st.button("Train Model"):
            df_preprocessed = preprocessor(df_flattened)
            st.write("Preprocessed Data:")
            st.write(df_preprocessed.head())
            model = get_model(model_name)

            try:
                predictions = train_model(model, df_preprocessed)

                st.write("Model Predictions:")
                st.write(predictions)

                if model_name in [
                    "DBSCAN",
                    "KMeans",
                ]:  # Add more clustering models if needed
                    clusters = predictions
                else:
                    clusters = None

                # Apply PCA for visualization
                pca_result, explained_variance = apply_pca(df_preprocessed)

                # Plot PCA results
                st.write("PCA Plot:")
                plot_pca(pca_result, explained_variance, clusters)

            except ValueError as e:
                st.error(f"An error occurred during model training: {e}")


if __name__ == "__main__":
    main()
