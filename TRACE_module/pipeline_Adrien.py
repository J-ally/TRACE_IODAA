
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:51:20 2025

@author: bouchet
"""

###############################################################################
#                                 IMPORTS                                     #
###############################################################################

import preprocessing as pp

import visualisation as vi 
from descriptive_analysis import from_distance_to_sequences_vector, from_distances_to_sequences_stack,from_seq_to_average_interaction_time,from_seq_to_daily_interactions

import numpy as np
import os 
import pandas as pd
import matplotlib.pyplot as plt

import dotenv
import sys
import time
from env_loading import parent_dir, output_dir

if os.getcwd().endswith("TRACE_module"):
    new_path = os.getcwd()[:-len("TRACE_module")] + "UTILS_module"
elif os.getcwd().endswith("TRACE_IODAA"):
    new_path = os.getcwd() + os.sep + "UTILS_module"
sys.path.insert(0, new_path) # insert path to be able to import file afterwards, allows for imports wherever the script is called from
    
from UTILS_files import test_import
if test_import() : # Testing the import
    from UTILS_files import get_all_files_within_one_folder # Importing the wanted function
else :
    print("Import failed")
    
    

import preprocessing as pp



###############################################################################
#                                 SCRIPTS                                     #
###############################################################################

folder= parent_dir
list_files=list()

##################################
#ETAPE 1 - Obtention des fichiers#
##################################


t1=time.perf_counter() 

list_files = get_all_files_within_one_folder(folder, True, extension=".parquet")

t2=time.perf_counter()
######################################
#ETAPE 2 - Concaténation des fichiers#
######################################


folder_savings =  os.sep.join([output_dir,"savings"])
if not  os.path.isdir(folder_savings) : 
    os.makedirs(folder_savings)

data=pp.concatenate_df(list_files)

##########################################
#ETAPE 3 -Transformation RSSI en distance#
##########################################
t3=time.perf_counter()

data=pp.transform_rssi_to_distance(data)
t4=time.perf_counter() 

###################################################################
#ETAPE 5 - Création d'un stack de matrice d'adjacence symétriques #
###################################################################

list_id=list(pd.unique(data["accelero_id"]))


stack,list_timesteps=pp.create_stack(data,list_id)
t5=time.perf_counter()



###################################################################
#ETAPE 6- Analyses                                                #
###################################################################


start_time = pd.Timestamp('2024-03-20T08:39:00.000000000')
end_time = pd.Timestamp('2024-04-10T16:36:00.000000000')

time_filter = (list_timesteps >= start_time) & (list_timesteps <= end_time)
filtered_timestamps = list_timesteps[time_filter]
# #cows_id = ['365d', '365e', '365f', '3662', '3663', '3664', '3665', '3666',
# #'3667', '3668', '3669', '366a', '366b', '366c', '366d', '3660']
# ### Analysis Pipeline ###
# cows_id=list_id
# ## Preprocessing of the RSSI matrix : time and symmetry
distances_time_cleaned = stack#[time_filter]
distances_clean=distances_time_cleaned
t5b=time.perf_counter() 
distances_clean=np.where(distances_clean==0,np.nan,distances_clean)
## Creation a a sequence matrix
matrice_seq = from_distances_to_sequences_stack(distances_clean)
t5c=time.perf_counter()

filtered_timestamps=list_timesteps
# calculation of the number of interaction each couple of cows had during the data collection
number_of_interaction_sequences = np.nanmax(np.nan_to_num(matrice_seq, nan=0), axis=0)
t5d=time.perf_counter() 
# calulate the number of daily interactions each cows has
number_of_daily_interaction = from_seq_to_daily_interactions(matrice_seq,filtered_timestamps)
t5e=time.perf_counter() 
# calculate the average time one interaction lasts for each cow
average_duration_of_an_interaction = from_seq_to_average_interaction_time(matrice_seq)

t6=time.perf_counter() 




## Plot the heatmap of the interactions number for each couple
plt.figure(figsize=(8, 6))
plt.imshow(number_of_interaction_sequences, cmap='hot', interpolation='nearest')
plt.colorbar(label="Maximum Sequence ID")
plt.title(" Between Cows")
plt.xlabel("Cow ID (Column)")
plt.ylabel("Cow ID (Row)")
plt.xticks(ticks=np.arange(len(list_id)), labels=list_id, rotation=45)
plt.yticks(ticks=np.arange(len(list_id)), labels=list_id)
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
ax.set_xticklabels(list_id)
ax.legend()

# Display the plot
plt.tight_layout()
plt.show() 


print("--BENCHMArk-- \n ==========\n étape 1 - localisation des données: {} \n étape 2 - concaténation des fichiers et création d'un dataframe unique ': {} \n étape 3 -transformation des signaux RSSI en distance {} \n étape 4 - Mise en forme des données sous forme matricielle: {} \n étape 5 : Analyse descriptive :{} ".format(t2-t1,t3-t2,t4-t3,t5-t4,t6-t5))


