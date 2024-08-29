# Avidea-Fraud-Redundancy

This app includes an algorithm that can detect redundancies across unique reports by selecting column combinations, as well as a data preprocessing pipeline and various unsupervised models to detect frauds.


# Getting Started:

## Prerequisites: 
- **Clone Repository:**
  ```bash
  git clone https://github.com/YassineNefzi/Avidea-Fraud-Redundancy.git
- **Install Dependencies:**
  ```bash
  pip install -r requirements.txt
- **Run Minerva:**
  ```bash
  streamlit run app.py

## Usage:

- ***Load CSV:*** Provide the path to your CSV file when prompted.
- ***Initial Analysis:*** The Algorithm will provide a first look at the data by displaying the head of the CSV file.
- ***Redundancy:*** Select the columns you want to be used for the combination analysis, also provide the column to group by then click on Process Redundancies. The algorithm will output a CSV file with the different column combinations and their unique report counts.
- ***Unsupervised Learning:*** Select one of the four available models :
    - KNN
    - DBSCAN
    - Isolation Forest
    - One-Class SVM </br>
    
After which you can click on Train model, the preprocessed dataframe will be displayed along with the PCA plot, as well as the potential fraudulent rows and their content in the original dataframe.
