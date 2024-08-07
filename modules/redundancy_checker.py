"""Module to check for redundant combinations of columns in a dataframe."""

import pandas as pd
from itertools import combinations


class RedundancyChecker:
    """
    Class to check for redundant combinations of columns in a dataframe.
    Performs the following steps:
    1. Preprocess the dataframe by filling missing values and dropping duplicates.
    2. Find redundant combinations of columns by grouping the dataframe by each combination.
    3. Filter and combine the results.
    4. Save the results to a CSV file.
    """

    def __init__(self, df, relevant_columns, group_column='report_id'):
        self.df = df
        self.relevant_columns = relevant_columns
        self.group_column = group_column

    def preprocess(self):
        self.df.fillna('-', inplace=True)
        self.df_unique = self.df.drop_duplicates()

    def find_redundant_combinations(self):
        results = []
        for i in range(len(self.relevant_columns), 1, -1):
            for combo in combinations(self.relevant_columns, i):
                grouped = self.df_unique.groupby(list(combo))[self.group_column].nunique().reset_index(name='unique_group_count')
                grouped = grouped[(grouped[list(combo)] != '-').all(axis=1)]
                grouped = grouped[grouped['unique_group_count'] > 2]
                grouped_sorted = grouped.sort_values(by='unique_group_count', ascending=False)
                grouped_sorted['combination_length'] = i
                if not grouped_sorted.empty:
                    results.append((combo, grouped_sorted))
        return results

    def filter_and_combine_results(self, results):
        combined_results = pd.DataFrame()
        for combo, grouped_sorted in results:
            grouped_sorted = grouped_sorted.dropna(how='all', subset=combo)
            if not grouped_sorted.empty:
                combined_results = pd.concat([combined_results, grouped_sorted], ignore_index=True)
        return combined_results

    def get_combined_redundancies(self):
        redundant_combinations = self.find_redundant_combinations()
        combined_redundancies = self.filter_and_combine_results(redundant_combinations)
        return combined_redundancies

    def save_results(self, output_path):
        combined_redundancies = self.get_combined_redundancies()
        combined_redundancies = combined_redundancies.sort_values(by=['combination_length', 'unique_group_count'], ascending=[False, False])
        combined_redundancies.to_csv(output_path, index=False)
