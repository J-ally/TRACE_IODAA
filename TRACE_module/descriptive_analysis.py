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



def from_distance_to_sequences_vector(vector, min_timesteps: int = 3, ignore_break: int = 2):
    """
    Function that takes a 1D numpy array of distances or RSSI signal for several
    timesteps and returns a vector annotated with sequence IDs.
    Allows to count the sequences and label each timestep

    Parameters:
    -----------
    vector : np.ndarray
        1D array of distances or signal values.
    min_timestep : int, optional
        Minimum number of consecutive timesteps to consider a sequence. Default is 3.
    ignore_break : int, optional
        Number of consecutive NaNs allowed before breaking a sequence. Default is 0.

    Returns:
    --------
    np.ndarray
        1D array with annotated sequence IDs.
    """
    sequence_vec = np.full_like(vector, np.nan, dtype=float)
    sequence_id = 1
    sequence_start = None
    nan_count = 0  # Tracks consecutive NaNs within the allowed wiggle room

    for i in range(len(vector)):
        if not np.isnan(vector[i]):  # If there is a valid distance
            if sequence_start is None:  # Start a new sequence if none active
                sequence_start = i
            nan_count = 0  # Reset NaN count when a valid value is encountered
        else:  # Encountered a NaN
            nan_count += 1
            if nan_count > ignore_break:  # Break the sequence if NaNs exceed the wiggle room
                if sequence_start is not None and i - nan_count - sequence_start >= min_timesteps:
                    sequence_vec[sequence_start:i - nan_count] = sequence_id  # Annotate sequence
                    sequence_id += 1
                sequence_start = None  # Reset the sequence start

    # Handle any remaining sequence at the end
    if sequence_start is not None and len(vector) - sequence_start >= min_timesteps:
        sequence_vec[sequence_start:] = sequence_id

    return sequence_vec

def from_distances_to_sequences_stack(stack, axis: int =0,min_timesteps : int =3, ignore_break=2):
    """
    Function that takes a stack of numpy array of distances or RSSI signal for several
    timesteps and returns a stack annotated with sequence IDs.
    Allows to count the sequences and label each timestep for each cow
    """

    matrice_seq = np.apply_along_axis(
    lambda x: from_distance_to_sequences_vector(x, min_timesteps=min_timesteps, ignore_break=ignore_break),
    axis=axis,
    arr=stack
    )
    return matrice_seq


def from_seq_to_average_interaction_time(matrice_seq, time_step :int =20):

    #Create a matrix that counts the total interactions :
    matrice_binary = np.where(np.isnan(matrice_seq), 0, 1)
    interaction_counts= np.sum(matrice_binary, axis=0) # 2D array cows x cows
    interaction_counts_cowwise = np.sum(interaction_counts, axis =0) # 1D array cows

    number_of_interaction_sequence = np.nansum(
        np.nanmax(np.nan_to_num(matrice_seq, nan=0), axis=0), axis=0) # 1D array cows


    total_time = interaction_counts_cowwise * time_step
    total_time_minutes = total_time / 60


    return total_time_minutes / number_of_interaction_sequence

def from_seq_to_daily_interactions(matrice_seq,filtered_timestamps) :
    unique_days = pd.to_datetime(filtered_timestamps).normalize().unique()
    num_days = len(unique_days)
    number_of_sequence = np.nanmax(np.nan_to_num(matrice_seq, nan=0), axis=0)
    daily_interactions_per_cow = np.nansum(number_of_sequence, axis=0) / num_days

    return daily_interactions_per_cow


def from_stack_to_number_of_interaction_sequences(
        matrice : np.ndarray
        ) -> np.ndarray: 
    """
    
    """
    return np.nanmax(np.nan_to_num(matrice, nan=0), axis=0)
