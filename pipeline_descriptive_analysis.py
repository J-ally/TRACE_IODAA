###############################################################################
#                                 IMPORTS                                     #
###############################################################################

from TRACE_module import preprocessing as pp
from TRACE_module import visualisation as vi
from TRACE_module import descriptive_analysis as da


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




###############################################################################
#                                 SCRIPTS                                     #
###############################################################################

####### SET FILES VARIABLES ##########
folder= 'data/data_rssi/20240319 - Cordemais/Data_rssi_glob_sensor_time'
list_files=list()
recolte = 'cordemais'

#COws we want to select the st for vis
sample_cows =

#### SET TIME VARIABLES FOR CROPPING######

if recolte == 'cordemais' :
    start_time = pd.Timestamp('2024-03-22')
    end_time = pd.Timestamp('2024-04-08')

elif recolte == 'buisson' :
    start_time = pd.Timestamp('2024-10-18')
    end_time = pd.Timestamp('2024-10-28')

elif recolte == 'blavet' :
    start_time = pd.Timestamp('2024-07-28')
    end_time = pd.Timestamp('2024-08-12')



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

#data=pp.transform_rssi_to_distance(data)
t4=time.perf_counter()

###################################################################
#ETAPE 5 - Création d'un stack de matrice d'adjacence symétriques #
###################################################################

list_id=list(pd.unique(data["accelero_id"]))


stack_raw,list_timesteps_raw =pp.create_stack(data,list_id)
stack, list_timesteps = da.sort_stack_by_timesteps(stack_raw,list_timesteps_raw)
t5=time.perf_counter()

###################################################################
#ETAPE 6- Analyses                                                #
###################################################################

distances_clean,list_timesteps=pp.crop_start_end_stack(stack=stack,
                         list_timesteps = list_timesteps ,
                         start = start_time,
                         end = end_time)

# ###




distances_clean=np.where(distances_clean==0,np.nan,distances_clean)
## Creation a a sequence matrix
matrice_seq = da.compute_neighbours_per_timestepfrom_distances_to_sequences_stack(distances_clean)

##Computing relevent number from the sequence matrix

number_of_interaction_sequences = da.from_stack_to_number_of_interaction_sequences(matrice_seq)
number_of_daily_interaction = da.from_seq_to_daily_interactions(matrice_seq,list_timesteps)
average_duration_of_an_interaction = da.from_seq_to_average_interaction_time(matrice_seq)

### Social env :

df_neighbors = da.compute_neighbours_per_timestep(stack= distances_clean,
                                timesteps= list_timesteps,
                                ids= list_id,
                                threshold= -65)

time_spent_df = da.compute_time_spent(df_neighbors)


# #####################
# #Visualisation#
# #####################


downloads_path = os.path.expanduser("~/Downloads")
folder_path = os.path.join(downloads_path, recolte)

# Create the folder if it does not exist
os.makedirs(folder_path, exist_ok=True)

figure_counter = 1  # Counter for naming figures

# Generate and save Heatmap
fig1 = vi.heatmap_interactions_number(number_of_interaction_sequences, list_id)
vi.save_figure(fig1, folder_path, figure_counter)
figure_counter += 1

# Generate and save Barplot (combined)
fig2 = vi.barplot_interaction_cows(number_of_daily_interaction, average_duration_of_an_interaction, list_id)
vi.save_figure(fig2, folder_path, figure_counter)
figure_counter += 1

# Generate and save Barplots (separate)
fig3 = vi.barplot_interaction_cows_separate(number_of_daily_interaction, average_duration_of_an_interaction, list_id)
vi.save_figure(fig3, folder_path, figure_counter)
figure_counter += 1

# Generate and save Boxplot (all around)
fig4 = vi.boxplot_average_time_number_interactions(number_of_daily_interaction, average_duration_of_an_interaction)
vi.save_figure(fig4, folder_path, figure_counter)
figure_counter += 1


# Barblot of the time spent with a certain number of neighbors for each cows
fig5 = da.plot_time_spent_with_neighbors(df_neighbors, place=recolte, use_percentage=True)
vi.save_figure(fig5, folder_path,figure_counter)
figure_counter += 1

#PLot for selected cows the time series of their number of neighbors
fig6 = da.plot_neighbors_ts(df_neighbors, cows= sample_cows)
vi.save_figure(fig6,folder_path,figure_counter)

print(f"\nAll figures saved in: {folder_path}")
