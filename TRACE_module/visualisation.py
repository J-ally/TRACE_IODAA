#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def date_to_index(date: str, list_date: pd.Series):
    return list_date[list_date == date].index[0]


def visualisation_distance_matrice(
    matrice: np.ndarray, labels: list[str], max_val: int, min_val: int = 0
) -> None:
    # Créer un masque pour les NaN
    mask = np.isnan(matrice)

    # Palette colorblind-friendly (cividis est perceptuellement uniforme)

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        matrice,
        annot=True,
        cmap="GnBu_r",  ## Pour Timothée qui est daltonien
        fmt=".2f",
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        linecolor="black",
        linewidths=1,
        mask=mask,
        vmin=min_val,  # Échelle min/max pour inclure toutes les valeurs non-NaN
        vmax=max_val,
    )

    # Inversion de l'échelle (palette claire pour valeurs élevées)
    plt.gca().invert_yaxis()  # Optionnelle : inverser l'axe Y pour alignement conventionnel

    # Ajout de la couleur grise pour les NaN
    plt.gca().set_facecolor("white")  # Couleur des cellules NaN

    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(rotation=45, fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_accelero_ranking(df: pd.DataFrame):

    """
    Fonction permettant d'afficher sous la forme d'un histogramme du classement des individus proches par animal

    **Args :
        df (Dataframe) : Dataframe contenant au moins les attributs "id_sensor" et "accelero_id" et un attribut de comptage (par exemple "count")

    **Returns : Histogramme
    """

    # Créer un graphique en barres groupées
    plt.figure(figsize=(10, 6))
    
    # Tracer les barres avec Seaborn
    sns.barplot(
        data=df,
        x='id_sensor',
        y='count',
        hue='accelero_id',  # Couleurs différentes pour chaque accelero_id
        dodge=True  # Pour séparer les barres par groupe
    )
    
    # Ajouter des labels et un titre
    plt.xlabel('ID Sensor')
    plt.ylabel('Nombre d\'occurrences')
    plt.title('Classement des individus proches par animal')
    plt.legend(title='Accelero ID')
    
    # Afficher le graphique
    plt.show()
