# post_clustering.py

import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def evaluate_clusters(data, categorical_columns, optimal_clusters):
    kproto = KPrototypes(n_clusters=optimal_clusters, init='Huang', random_state=42)
    clusters = kproto.fit_predict(data, categorical=[data.columns.get_loc(col) for col in categorical_columns])
    
    # Calculer le coefficient de silhouette en utilisant les données encodées
    silhouette_avg = silhouette_score(data, clusters, metric='euclidean')
    print(f'For n_clusters = {optimal_clusters}, the silhouette score is {silhouette_avg}')
    
    return clusters

def interpret_clusters(data, clusters):
    # Ajouter les clusters au DataFrame
    data['Cluster'] = clusters
    
    # Agréger les données par cluster et calculer les statistiques
    cluster_summary = data.groupby('Cluster').agg(['mean', 'median', 'count'])
    print("\nCluster Summary:")
    print(cluster_summary)

    # Interpréter chaque cluster
    for cluster_num in range(data['Cluster'].nunique()):
        print(f"\nInterprétation du cluster {cluster_num}:")
        cluster_data = data[data['Cluster'] == cluster_num]
        print(cluster_data.describe())
        print(cluster_data['PRIMTOTALE'].describe())

if __name__ == "__main__":
    file_path = 'Processed_Dataset_clients.csv'
    categorical_columns = ['VILLE', 'PROFESSION']
    
    