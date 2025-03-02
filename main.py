import os
from src.data_processing import load_data, preprocess_data

DATA_DIR = 'data'

def main():
    file_path = os.path.join(DATA_DIR, 'sales_data.csv')
    df = load_data(file_path)

    if df is not None:
        print("Raw Data:")
        print(df.head())

        df = preprocess_data(df)
        print("Processed Data:")
        print(df.head())

if __name__ == "__main__":
    main()
