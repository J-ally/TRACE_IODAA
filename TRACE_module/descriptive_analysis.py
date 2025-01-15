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
        #filter_date: str = 'date',  #or 'hour'
        limit_rank: int = 10,
        ) -> pd.DataFrame:
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
    
    # Ajout de l'attribut "date" de la campagne de relevé des capteurs RSSI
    filtered_df["date"] = filtered_df['glob_sensor_DateTime'].dt.date
    
    #Ajout de l'attribut "time_diff" qui calcule le temps d'écart entre le dernier temps de prise de mesure et celui actuellement considéré
    filtered_df['time_diff'] = filtered_df['glob_sensor_DateTime'].diff().dt.total_seconds()

    #Création de séquence lorsque la paire de capteur change ou lorsque l'intéraction n'est pas
    #considérée comme continue (> 40 sec)
    filtered_df['sequence'] = (
    (filtered_df['id_sensor'] != filtered_df['id_sensor'].shift()) | # Vérifie les changements dans id_sensor
    (filtered_df['accelero_id'] != filtered_df['accelero_id'].shift()) |
    (filtered_df['time_diff'] > 40.0)
    ).cumsum()

    # On met toutes les durées à 20 secondes pour les calculs de temps totaux et moyens d'intéractions
    filtered_df["time_diff_real"] = 20.0

    # Somme des time_diff et nombre de séquences et d'intéractions par jour
    metrics_analysis = (
    filtered_df
    .groupby(['date', 'id_sensor', 'accelero_id'], as_index=False)
    .agg(
        nb_interactions=('sequence', 'size'),  # Compter les occurrences
        total_time=('time_diff_real', 'sum'), # Somme de time_diff_real
        len_seq=('sequence', 'nunique')) # Nombre de séquences uniques
    )
    metrics_analysis['average_time_min'] = (metrics_analysis['total_time']/60) / metrics_analysis['len_seq']
    metrics_analysis['total_time_min'] = metrics_analysis['total_time']/60

    # # Trier les résultats pour chaque "id_sensor" par ordre décroissant de "count"
    # sorted_metrics = metrics_analysis.sort_values(
    #     by=['date', 'id_sensor', 'average_time_min'], 
    #     ascending=[True, True, False])
    # # Limiter le nombre de "accelero_id" par "id_sensor"
    # top_metrics = sorted_metrics.groupby(['id_sensor']).head(limit_rank)
    return metrics_analysis
