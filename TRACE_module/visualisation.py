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
