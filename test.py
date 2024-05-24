# Import module for data manipulation
import pandas as pd
# Import module for linear algebra
import numpy as np
# Import module for data visualization
from plotnine import * 
import plotnine
# Import module for k-protoype cluster
from kmodes.kprototypes import KPrototypes
# Ignore warnings
import warnings
warnings.filterwarnings('ignore', category = FutureWarning)
# Format scientific notation from Pandas
pd.set_option('display.float_format', lambda x: '%.3f' % x)

## ######## Chargement des donnees ##########
df = pd.read_csv('Dataset_clients_sanlam.csv', sep=';', encoding='iso-8859-1')
# La dimension des données
print('Dimension data: {} rows and {} columns'.format(len(df), len(df.columns)))
# Imprimer les 5 premières lignes
print(df.head())

####### Inspecter le type de données #########
print(df.info())
####### Changement du type de donnees de la colone PRIMTOTALE en entier #########

df['PRIMTOTALE'] = df['PRIMTOTALE'].astype(float)
df['PRIMNETTE'] = df['PRIMNETTE'].astype(float)

# Supprimer les lignes avec des valeurs non finies dans la colonne 'CODECAT'
df = df.dropna(subset=['CODECAT'])

# Convertir la colonne 'CODECAT' en entiers
df['CODECAT'] = df['CODECAT'].astype(int)
df.info()
print(df.info())

# Inspectez les variables catégorielles
df.select_dtypes( 'object' ).nunique()
print(df.select_dtypes( 'object' ).nunique())

# Inspectez les variables numériques
df.describe()
print(df.describe())
# Vérifiez la valeur manquante
df.isna().sum ()
print(df.isna().sum())

###########################Exploration des donnees########################################################
# Répartition des clients par ville
df_ville = pd.DataFrame(df['VILLE'].value_counts()).reset_index()
df_ville['Percentage'] = df_ville['VILLE'] / df['VILLE'].value_counts().sum()
df_ville.rename(columns={'index': 'Ville', 'VILLE': 'Total'}, inplace=True)
df_ville = df_ville.sort_values('Total', ascending=True).reset_index(drop=True)

# La trame de données pour la répartition par ville
df_ville_details = df.groupby('VILLE').agg({
    'VILLE': 'count',
    'PRIMNETTE': 'sum',
    'PRIMTOTALE': 'sum',
    'AVENANT': 'mean',
    'AGE': 'mean'
}).rename(columns={'VILLE': 'Total'}).reset_index().sort_values('Total', ascending=True)

print("Répartition des clients par ville:")
print(df_ville)
print("\nDétails de la répartition par ville:")
print(df_ville_details)

####### Répartition des clients par profession#######
df_profession = pd.DataFrame(df['PROFESSION'].value_counts()).reset_index()

df_profession['Percentage'] = df_profession['PROFESSION'].value_counts() / df['PROFESSION'].value_counts().sum()
df_profession.rename(columns={'index': 'Profession', 'PROFESSION': 'Total'}, inplace=True)
df_profession = df_profession.sort_values('Total', ascending=True).reset_index(drop=True)

# La trame de données pour la répartition par profession
df_profession_details = df.groupby('PROFESSION').agg({
    'PROFESSION': 'count',
    'PRIMNETTE': 'sum',
    'PRIMTOTALE': 'sum',
    'AVENANT': 'mean',
    'AGE': 'mean'
}).rename(columns={'PROFESSION': 'Total'}).reset_index().sort_values('Total', ascending=True)

print("\nRépartition des clients par profession:")
print(df_profession)
print("\nDétails de la répartition par profession:")
print(df_profession_details)

#######
# Créer des groupes d'âge
bins = [0, 30, 40, 50, 60, 100]
labels = ['<30', '30-40', '41-50', '51-60', '60+']
df['AgeGroup'] = pd.cut(df['AGE'], bins=bins, labels=labels, right=False)

# Répartition des clients par groupe d'âge
df_agegroup = pd.DataFrame(df['AgeGroup'].value_counts()).reset_index()
df_agegroup['Percentage'] = df_agegroup['AgeGroup'].value_counts() / df['AgeGroup'].value_counts().sum()
df_agegroup.rename(columns={'index': 'AgeGroup', 'AgeGroup': 'Total'}, inplace=True)
df_agegroup = df_agegroup.sort_values('Total', ascending=True).reset_index(drop=True)

# La trame de données pour la répartition par groupe d'âge
df_agegroup_details = df.groupby('AgeGroup').agg({
    'AgeGroup': 'count',
    'PRIMNETTE': 'sum',
    'PRIMTOTALE': 'sum',
    'AVENANT': 'mean',
    'AGE': 'mean'
}).rename(columns={'AgeGroup': 'Total'}).reset_index().sort_values('Total', ascending=True)

print("\nRépartition des clients par groupe d'âge:")
print(df_agegroup)
print("\nDétails de la répartition par groupe d'âge:")
print(df_agegroup_details)


########### Créer des groupes d'âge###########
bins = [0, 30, 40, 50, 60, 100]
labels = ['<30', '30-40', '41-50', '51-60', '60+']
df['AgeGroup'] = pd.cut(df['AGE'], bins=bins, labels=labels, right=False)

# Répartition des clients par groupe d'âge
df_agegroup = pd.DataFrame(df['AgeGroup'].value_counts()).reset_index()
df_agegroup['Percentage'] = df_agegroup['AgeGroup'].value_counts() / df['AgeGroup'].value_counts().sum()
df_agegroup.rename(columns={'index': 'AgeGroup', 'AgeGroup': 'Total'}, inplace=True)
df_agegroup = df_agegroup.sort_values('Total', ascending=True).reset_index(drop=True)

# La trame de données pour la répartition par groupe d'âge
df_agegroup_details = df.groupby('AgeGroup').agg({
    'AgeGroup': 'count',
    'PRIMNETTE': 'sum',
    'PRIMTOTALE': 'sum',
    'AVENANT': 'mean',
    'AGE': 'mean'
}).rename(columns={'AgeGroup': 'Total'}).reset_index().sort_values('Total', ascending=True)

print("\nRépartition des clients par groupe d'âge:")
print(df_agegroup)
print("\nDétails de la répartition par groupe d'âge:")
print(df_agegroup_details)
###########################################################
# Graphique de répartition par profession
plot_profession = (
    ggplot(data = df_profession) +
    geom_bar(aes(x = 'PROFESSION', y = 'Total'),
             fill = '#80797c',
             stat = 'identity') +
    geom_text(aes(x = 'PROFESSION', y = 'Total', label = 'Total'), size = 10, nudge_y = 120) +
    labs(title = 'Répartition des clients par profession') +
    xlab('PROFESSION') +
    ylab('Nombre de clients') +
    scale_x_discrete(limits = df_profession['PROFESSION'].tolist()) +
    theme_minimal() +
    coord_flip()
)
print(plot_profession)
