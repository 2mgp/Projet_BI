# exploration.py

import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path, sep=';')
    return data

def describe_data(data):
    print("Description des données :")
    print(data.describe())
    print("\nInformations sur les données :")
    print(data.info())
    print("\nPremières lignes des données :")
    print(data.head())

if __name__ == "__main__":
    file_path = 'Dataset_clients.csv'
    data = load_data(file_path)
    describe_data(data)
