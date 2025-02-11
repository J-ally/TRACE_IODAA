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



def sort_stack_by_timesteps(stack: np.ndarray, list_timesteps: np.array) -> tuple[np.ndarray, np.array]:
    """
    Sorts the stack and timesteps to ensure they are in chronological order.

    Args:
    - stack (np.ndarray): A 3D NumPy array of shape (T, N, N), where T is the number of timesteps.
    - list_timesteps (np.array): A 1D NumPy array of shape (T,) containing unsorted timestamps.

    Returns:
    - tuple: (sorted_stack, sorted_list_timesteps)
    """

    # Get sorted indices based on timestamps
    sorted_indices = np.argsort(list_timesteps)

    # Sort both stack and list_timesteps using the same order
    sorted_stack = stack[sorted_indices]
    sorted_list_timesteps = list_timesteps[sorted_indices]

    return sorted_stack, sorted_list_timesteps


def from_distance_to_sequences_vector(vector, min_timesteps: int = 3, ignore_break: int = 2):
    """
    Function that takes a 1D numpy array of distances or RSSI signal for several
    timesteps and returns a vector annotated with sequence IDs.
    Allows to count the sequences and label each timestep.

    Parameters:
    -----------
    vector : np.ndarray
        1D array of distances or signal values.
    min_timestep : int, optional
        Minimum number of consecutive timesteps to consider a sequence. Default is 3.
    ignore_break : int, optional
        Number of consecutive NaNs or 0 allowed before breaking a sequence. Default is 2.

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
    #TODO
    # Optimize the performance, quite slow : should be farily simple.
    return sequence_vec

def from_distances_to_sequences_stack(stack, axis: int =0,min_timesteps : int =3, ignore_break=2):
    """
    Function that takes a 3D numpy array of distances or RSSI signal for several
    timesteps and returns a 3D numpy array annotated with sequence IDs.
    Allows to count the sequences and label each timestep.
    Applies the function from_distances_to_sequences_vector in the whole stack

    Parameters:
    -----------
    stack : np.ndarray
        3D array of distances or signal values.
    axis : int
        Axis for which the sequences are represented. Default is 0, according to our data.
    min_timestep : int, optional
        Minimum number of consecutive timesteps to consider a sequence. Default is 3.
    ignore_break : int, optional
        Number of consecutive NaNs or 0 allowed before breaking a sequence. Default is 2.

    Returns:
    --------
    np.ndarray
        3D array with annotated sequence IDs.
    """

    matrice_seq = np.apply_along_axis(
    lambda x: from_distance_to_sequences_vector(x, min_timesteps=min_timesteps, ignore_break=ignore_break),
    axis=axis,
    arr=stack
    )
    return matrice_seq

def from_stack_to_number_of_interaction_sequences(
        matrice : np.ndarray
        ) -> np.ndarray:
    """
    Function that takes the array with annotated sequences for each timestep
    (from from_distances_to_sequences_stack) and extracts the number of interactions
    """
    return np.nanmax(np.nan_to_num(matrice, nan=0), axis=0)

def from_seq_to_average_interaction_time(matrice_seq, time_step :int =20):
    """
    Function that computes the average interaction time for each cow in minutes.

    Parameters:
    -----------
    matrice_seq : np.ndarray
        3D array with the sequences annotated for each timestep.
        Output of from_distances_to_sequences_stack
    time_step : int,
        Duration of one timestep = the parameter used in the smoothing operation during the preprocessing.

    Returns:
    --------
    np.ndarray
        1D array with an average interaction time for each cow.
    """
    #Create a matrix that counts the total time cows interacted within an interaction:
    matrice_binary = np.where(np.isnan(matrice_seq), 0, 1)
    interaction_counts= np.sum(matrice_binary, axis=0) # 2D array cows x cows
    interaction_counts_cowwise = np.sum(interaction_counts, axis =0) # 1D array cows

    #Counts the number of interaction for each cows
    number_of_interaction_sequence = np.nansum(
        from_stack_to_number_of_interaction_sequences(matrice_seq),
        axis=0) # 1D array cows

    # Replace zeros with NaN to avoid division errors
    number_of_interaction_sequence[number_of_interaction_sequence == 0] = np.nan

    #Averages the duration of an interaction
    total_time = interaction_counts_cowwise * time_step
    total_time_minutes = total_time / 60
    print (total_time_minutes)
    print(number_of_interaction_sequence)
    return total_time_minutes / number_of_interaction_sequence

def from_seq_to_daily_interactions(matrice_seq,filtered_timestamps) :
    """"
    Function that computes the average interaction time for each cow.
    Sums all of the interaction one cow has with every other one.
    Parameters:
    -----------
    matrice_seq : np.ndarray
        3D array with the sequences annotated for each timestep.
        Output of from_distances_to_sequences_stack
    filtered_timestamps : list
        list of pd.TimeStamps corresponding to the timestamps from the matrice_seq

    Returns:
    --------
    np.ndarray
        1D array with an average interaction time for each cow.
    """
    # Counts the number of days in the data
    unique_days = pd.to_datetime(filtered_timestamps).normalize().unique()
    num_days = len(unique_days)

    # Number of interactions for each couple of cows
    number_of_sequence = from_stack_to_number_of_interaction_sequences(matrice_seq)

    #Number of interaction cowwise averaged for the total duration
    daily_interactions_per_cow = np.nansum(number_of_sequence, axis=0) / num_days

    return daily_interactions_per_cow

### Social environment analysis ###

def compute_neighbors_per_timestep(stack: np.ndarray, timesteps: list, ids: list, threshold: float) -> pd.DataFrame:
    """
    Converts a stack of matrices into adjacency matrices using a threshold, then computes the number of neighbors
    for each ID at each timestep.

    Parameters:
    -----------
    stack : np.ndarray
        3D array (T x N x N), where T is the number of timesteps, and N is the number of IDs.
    timesteps : list
        List of timesteps corresponding to each matrix.
    ids : list
        List of IDs corresponding to rows/columns of matrices.
    threshold : float
        Value above which two elements are considered connected (binary adjacency matrix).

    Returns:
    --------
    pd.DataFrame
        DataFrame where rows are timesteps, columns are IDs, and values are the number of neighbors per ID.
    """
    # Check consistency
    assert stack.shape[0] == len(timesteps), "Mismatch between stack and timesteps length"
    assert stack.shape[1] == stack.shape[2] == len(ids), "Mismatch between matrix size and IDs length"

    # Dictionary to store results
    neighbors_data = {id_: [] for id_ in ids}

    for t_idx, timestep in enumerate(timesteps):
        # Convert to adjacency matrix using threshold
        adjacency_matrix = (stack[t_idx] > threshold).astype(int)
        np.fill_diagonal(adjacency_matrix, 0)  # No self-connections

        # Compute neighbors (sum of ones per row)
        neighbors_count = np.sum(adjacency_matrix, axis=1)

        # Store results
        for id_idx, id_ in enumerate(ids):
            neighbors_data[id_].append(neighbors_count[id_idx])

    # Create DataFrame with timesteps as index
    df_neighbors = pd.DataFrame(neighbors_data, index=timesteps)

    return df_neighbors

def compute_time_spent_with_neighbors(df_neighbors: pd.DataFrame) -> pd.DataFrame:
    time_spent = {}
    """
    Converts the number of neighbors for each ID at each timesteps into a DataFrame that has the time spent with a
    certain number of neighbours.

    Parameters:
    -----------
   df_neighbours : pd.DataFrame
        DataFrame where rows are timesteps, columns are IDs, and values are the number of neighbors per ID.

    Returns:
    --------
    pd.DataFrame
        DataFrame where rows are timesteps, columns are IDs, and values are the number of neighbors per ID.
    """

    for cow_id in df_neighbors.columns:
        # Count occurrences of each neighbor count for the cow
        neighbor_counts = df_neighbors[cow_id].value_counts().sort_index()

        # Aggregate time spent in hours
        time_spent[cow_id] = (neighbor_counts * 20) / 3600

        # Merge all values >=5 into a "5+ neighbors" category
        if any(neighbor_counts.index >= 5):
            time_spent[cow_id].loc[5] = time_spent[cow_id].loc[neighbor_counts.index >= 5].sum()
            time_spent[cow_id] = time_spent[cow_id].drop(neighbor_counts.index[neighbor_counts.index > 5], errors="ignore")

    # Convert to DataFrame, fill missing values with 0, and drop the 0 neighbors category
    return pd.DataFrame(time_spent).fillna(0).drop(index=0, errors="ignore").sort_index()

def plot_time_spent_with_neighbors(df_neighbors: pd.DataFrame, place: str, use_percentage: bool=True):
    """
    Function to visualize time spent with neighbors in either absolute hours or percentage of data collection duration.

    Parameters:
    df_neighbors (DataFrame): Time series data of neighbor counts per cow.
    place (str): Location title for the plot.
    use_percentage (bool): If True, plots time in percentage, otherwise in absolute hours.

    Returns:
    fig (matplotlib.figure.Figure): The generated figure object.
    """
    # Compute time spent per number of neighbors (in hours, excluding 0)
    time_spent_df = compute_time_spent_with_neighbors(df_neighbors)

    # Compute total data collection duration in hours
    data_collection_duration = (df_neighbors.index[-1] - df_neighbors.index[0]).total_seconds() / 3600

    # Compute percentage of total data collection time for each cow
    if use_percentage:
        time_spent_df = (time_spent_df / data_collection_duration) * 100

    # Number of cows
    num_cows = len(df_neighbors.columns)

    # Set up a 4-column grid for the plots (and as many rows as needed)
    n_cols = 4
    n_rows = (num_cows // n_cols) + (1 if num_cows % n_cols != 0 else 0)

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 5))
    fig.suptitle(f"Time Spent with Friends (Percentage) - {place}", fontsize=16)

    # Flatten axes array for easy indexing
    axes = axes.flatten()

    # Plot for each cow in the grid
    for i, cow_id in enumerate(df_neighbors.columns):
        time_spent_df[cow_id].plot(kind="bar", color="cornflowerblue", ax=axes[i])
        axes[i].set_xlabel("Number of Neighbors")
        axes[i].set_ylabel("Percentage of Data Collection Duration (%)" if use_percentage else "Time Spent (hours)")
        axes[i].set_title(f"Cow {cow_id}")
        axes[i].tick_params(axis='x', rotation=0)
        axes[i].grid(axis="y", linestyle="--", alpha=0.7)

    # Hide any unused axes (if the number of cows is not a perfect multiple of 4)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def plot_neighbors_ts(df_neighbors:pd.DataFrame, cows:list, opacity:float=0.4):
    """
    Function to plot the number of neighbors over time for selected cows, grouped by day.

    Parameters:
    df_neighbors (DataFrame): Time series data of neighbor counts per cow.
    cows (list): List of cow IDs to plot.
    opacity (float) : opacity of the dots plotted for the time serie.

    Returns:
    figs (list): List of generated figure objects.
    """
    figs = []

    # Ensure index is in datetime format
    df_neighbors.index = pd.to_datetime(df_neighbors.index)

    # Group by day
    daily_groups = df_neighbors.groupby(df_neighbors.index.date)

    # Plot for each day
    for day, daily_data in daily_groups:
        fig, ax = plt.subplots(figsize=(12, 6))

        for cow in cows:
            ax.scatter(daily_data.index, daily_data[cow], label=f'Cow {cow}', alpha=opacity)

        ax.set_xlabel("Time")
        ax.set_ylabel("Number of Neighbors")
        ax.set_title(f"Number of Neighbors Over Time - {day}")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)

        # Format x-axis to show only hours
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

        figs.append(fig)

    return figs
