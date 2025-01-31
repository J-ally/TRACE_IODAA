### Script for descriptive analysis ###

import pandas as pd
import pyarrow
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append("/Users/elise/code/TRACE_IODAA/TRACE_module")
import elise_descriptive_analysis as desc
import preprocessing as prep
import numpy as np


#### FONCTIONS A METTRE DANS PREPROC ET DESCR ANALYSIS ###

def symetrize_3d_matrix(matrice_3d, type_="RSSI"):
    """
    Apply the function of symmetry for a whole matrix stack
    """
    # Créer une matrice résultat avec la même forme
    matrice_res_3d = np.zeros_like(matrice_3d)

    # Parcourir chaque matrice (16, 16) dans la dimension 0
    for t in range(matrice_3d.shape[0]):
        matrice_res_3d[t] = prep.symetrisation_matrice(matrice_3d[t], type_=type_)

    return matrice_res_3d


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

def from_seq_to_daily_interactions(matrice_seq) :
    unique_days = pd.to_datetime(filtered_timestamps).normalize().unique()
    num_days = len(unique_days)
    number_of_sequence = np.nanmax(np.nan_to_num(matrice_seq, nan=0), axis=0)
    daily_interactions_per_cow = np.nansum(number_of_sequence, axis=0) / num_days

    return daily_interactions_per_cow

### VARIABLES ###

distances = np.load('data/matrix_Cornemais/matrice_test.npy')
list_timesteps = np.load('data/matrix_Cornemais/list_timesteps.npy')
timestep = 20

start_time = pd.Timestamp('2024-03-20T08:39:00.000000000')
end_time = pd.Timestamp('2024-04-10T16:36:00.000000000')

time_filter = (list_timesteps >= start_time) & (list_timesteps <= end_time)
filtered_timestamps = list_timesteps[time_filter]
cows_id = ['365d', '365e', '365f', '3662', '3663', '3664', '3665', '3666',
'3667', '3668', '3669', '366a', '366b', '366c', '366d', '3660']

### Analysis Pipeline ###

## Preprocessing of the RSSI matrix : time and symmetry
distances_time_cleaned = distances[time_filter]
distances_clean = symetrize_3d_matrix(distances_time_cleaned)

## Creation a a sequence matrix
matrice_seq = from_distances_to_sequences_stack(distances_clean)

# calculation of the number of interaction each couple of cows had during the data collection
number_of_interaction_sequences = np.nanmax(np.nan_to_num(matrice_seq, nan=0), axis=0)

# calulate the number of daily interactions each cows has
number_of_daily_interaction = from_seq_to_daily_interactions(matrice_seq)

# calculate the average time one interaction lasts for each cow
average_duration_of_an_interaction = from_seq_to_average_interaction_time(matrice_seq)

### PLOTTING ###

## Plot the heatmap of the interactions number for each couple
plt.figure(figsize=(8, 6))
plt.imshow(number_of_interaction_sequences, cmap='hot', interpolation='nearest')
plt.colorbar(label="Maximum Sequence ID")
plt.title(" Between Cows")
plt.xlabel("Cow ID (Column)")
plt.ylabel("Cow ID (Row)")
plt.xticks(ticks=np.arange(len(cows_id)), labels=cows_id, rotation=45)
plt.yticks(ticks=np.arange(len(cows_id)), labels=cows_id)
plt.show()

## PLotting the double boxplot :

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 6))
# Bar width and x-axis positions
bar_width = 0.35
x = np.arange(1, 17)  # Cow indices (1 to 16)

# Plotting the bars
bars1 = ax.bar(x - bar_width / 2, number_of_daily_interaction, bar_width, label='Avg Daily Interactions')
bars2 = ax.bar(x + bar_width / 2, average_duration_of_an_interaction, bar_width, label='Avg Interaction Time (mins)')

# Labeling the chart
ax.set_xlabel('Cow Index')
ax.set_ylabel('Values')
ax.set_title('Comparison of Average Daily Interactions and Average Interaction Time for Each Cow')
ax.set_xticks(x)
ax.set_xticklabels(cows_id)
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()
