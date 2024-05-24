# preprocessing.py

import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path, sep=';')
    return data

def preprocess_data(data, categorical_columns):
    for col in categorical_columns:
        data[col] = data[col].astype('category')
        data[col] = data[col].cat.codes
    return data

if __name__ == "__main__":
    file_path = 'Dataset_clients.csv'
    categorical_columns = ['VILLE', 'PROFESSION']
    data = load_data(file_path)
    processed_data = preprocess_data(data, categorical_columns)
    processed_data.to_csv('Processed_Dataset_clients.csv', index=False)
