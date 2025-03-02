import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file into a Pandas DataFrame.
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found at {file_path}. Please add your dataset.")
        return None

def preprocess_data(df):
    """
    Clean and preprocess the data.
    - Removes rows with missing values.
    - (Placeholder for future data transformations)
    """
    df = df.dropna()  # Remove missing values (NaN)
    return df
