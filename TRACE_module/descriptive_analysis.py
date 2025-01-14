#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm

def rank_accelero_by_sensor(
        df: pd.DataFrame,
        threshold: int = -40,
        filter_date: str = 'date', #or 'hour'
        limit_rank: int = 10,
        
        )-> pd.DataFrame:
    """
    Fonction permettant de faire un classement des binômes de vaches qui passent le plus de temps proches 
    (lorsqu'il y a détection des capteurs RSSI) et de manière continue

    **Args :
        df (Dataframe) : Dataframe contenant au moins les attributs "id_sensor" et "accelero_id" et lissé selon un pas de temps
        limit_rank (int) : Paramètre permettant de limiter les rangs à afficher. Si on souhaite avoir le top 3 ou top 10 par exemple
        threshold (int) : Seuil de distance RSSI choisis

    **Returns : 
        pd.DataFrame : Dataframe avec les attributs "id_sensor" et "accelero_id" ainsi que les attributs du temps cumulé passé par binôme de vaches
        et par jour "count"
    """
    # Filtrer les lignes où "id_sensor" et "accelero_id" sont différents
    filtered_df = df[df['id_sensor'] != df['accelero_id']]
    filtered_df = filtered_df[filtered_df['RSSI'] >= threshold]

    filtered_df["date"] = filtered_df['glob_sensor_DateTime'].dt.date

    
    # Compter les occurrences de "accelero_id" pour chaque "id_sensor"
    counts = (filtered_df
              .groupby(['id_sensor', 'accelero_id'])
              .size()  # Compte les occurrences
              .reset_index(name='count')  # Renomme la colonne de comptage
              )

    # Trier les résultats pour chaque "id_sensor" par ordre décroissant de "count"
    sorted_counts = counts.sort_values(by=['id_sensor', 'count'], ascending=[True, False])
    
    # Limiter le nombre de "accelero_id" par "id_sensor"
    top_counts = sorted_counts.groupby('id_sensor').head(limit_rank)
    
    return top_counts