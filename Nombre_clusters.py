import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Chargement des données
try:
    data = pd.read_csv('Dataset_clients.csv', sep=';')
    print("Données chargées avec succès.")
    print(data.columns)
except Exception as e:
    print(f"Erreur lors du chargement des données : {e}")
    exit()

# Conversion des colonnes catégorielles en types appropriés
categorical_columns = ['VILLE', 'PROFESSION']  # Remplacez par les noms de vos colonnes catégorielles
try:
    for col in categorical_columns:
        data[col] = data[col].astype('category')
    print("Colonnes catégorielles converties avec succès.")
except Exception as e:
    print(f"Erreur lors de la conversion des colonnes catégorielles : {e}")
    exit()

# Préparation des données pour K-prototypes
try:
    for col in categorical_columns:
        data[col] = data[col].cat.codes
    print("Colonnes catégorielles encodées avec succès.")
except Exception as e:
    print(f"Erreur lors de l'encodage des colonnes catégorielles : {e}")
    exit()

# Déterminer le nombre de clusters optimal avec la méthode du coude
cost = []
try:
    for num_clusters in range(1, 10):
        kproto = KPrototypes(n_clusters=num_clusters, init='Huang', random_state=42)
        kproto.fit_predict(data, categorical=[data.columns.get_loc(col) for col in categorical_columns])
        cost.append(kproto.cost_)
    print("Calcul des coûts terminé.")
except Exception as e:
    print(f"Erreur lors du calcul des coûts : {e}")
    exit()

# Plotting the elbow method
try:
    plt.plot(range(1, 10), cost, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Cost')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    print("Méthode du coude tracée avec succès.")
except Exception as e:
    print(f"Erreur lors de la méthode du coude : {e}")
    exit()

# Calculer et afficher le coefficient de silhouette pour différents nombres de clusters
try:
    for num_clusters in range(2, 10):
        kproto = KPrototypes(n_clusters=num_clusters, init='Huang', random_state=42)
        clusters = kproto.fit_predict(data, categorical=[data.columns.get_loc(col) for col in categorical_columns])
        silhouette_avg = silhouette_score(data, clusters, metric='euclidean')
        print(f'For n_clusters = {num_clusters}, the silhouette score is {silhouette_avg}')
    print("Calcul du coefficient de silhouette terminé.")
except Exception as e:
    print(f"Erreur lors du calcul du coefficient de silhouette : {e}")
    exit()
