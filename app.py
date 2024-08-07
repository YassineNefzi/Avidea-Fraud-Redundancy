import streamlit as st
import pandas as pd
from io import StringIO
from utils.preprocessor import process_file, flatten_dataframe

def main():
    st.title("Fraud Detection App")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        df_flattened = flatten_dataframe(df)

        st.write("Flattened DataFrame:")
        st.write(df_flattened.head())

        relevant_columns = st.multiselect(
            "Select Columns for Combination Analysis",
            options=df_flattened.columns.tolist()
        )

        group_column = st.selectbox(
            "Select Column to Group By",
            options=df_flattened.columns.tolist(),
            index=df_flattened.columns.tolist().index('report_id') if 'report_id' in df_flattened.columns else 0
        )

        if st.button("Process"):
            if not relevant_columns:
                st.error("Please select at least one column.")
            else:
                output_file = process_file(df_flattened, relevant_columns, group_column)
                st.success(f"Redundancy analysis completed. Download the result file below.")
                
                with open(output_file, "rb") as file:
                    st.download_button(
                        label="Download CSV",
                        data=file,
                        file_name="combined_redundancies_across_reports.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()