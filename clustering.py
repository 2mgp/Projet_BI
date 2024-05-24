# clustering.py

import pandas as pd
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def find_optimal_clusters(data, categorical_columns):
    cost = []
    for num_clusters in range(1, 10):
        kproto = KPrototypes(n_clusters=num_clusters, init='Huang', random_state=42)
        kproto.fit_predict(data, categorical=[data.columns.get_loc(col) for col in categorical_columns])
        cost.append(kproto.cost_)

    # Plotting the elbow method
    plt.plot(range(1, 10), cost, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Cost')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    
    # Return the optimal number of clusters
    optimal_clusters = cost.index(min(cost[1:])) + 1  # Ignoring the first cost value as it's for 1 cluster
    return optimal_clusters

if __name__ == "__main__":
    file_path = 'Processed_Dataset_clients.csv'
    categorical_columns = ['VILLE', 'PROFESSION']
    data = load_data(file_path)
    optimal_clusters = find_optimal_clusters(data, categorical_columns)
    print(f'Optimal number of clusters: {optimal_clusters}')
