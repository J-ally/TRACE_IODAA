###############################################################################
#                                 IMPORTS                                     #
###############################################################################

import preprocessing as pp

import visualisation as vi
from descriptive_analysis import from_stack_to_number_of_interaction_sequences,from_distance_to_sequences_vector, from_distances_to_sequences_stack,from_seq_to_average_interaction_time,from_seq_to_daily_interactions

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

####### SET FILES VARIABLES ##########
folder= '/Users/elise/code/TRACE_IODAA/data/data_rssi/Data_rssi_glob_sensor_time_blavet'
list_files=list()

#### SET TIME VARIABLES FOR CROPPING######

### For Cordemais :
# start_time = pd.Timestamp('2024-03-22')
# end_time = pd.Timestamp('2024-04-08')

### For Buisson :
#start_time = pd.Timestamp('2024-10-17')
#end_time = pd.Timestamp('2024-10-29')

### For Blavet :
start_time = pd.Timestamp('2024-07-27')
end_time = pd.Timestamp('2024-08-13')



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


stack,list_timesteps=pp.create_stack(data,list_id)
t5=time.perf_counter()

###################################################################
#ETAPE 6- Analyses                                                #
###################################################################

distances_clean,list_timesteps=pp.crop_start_end_stack(stack=stack,
                         list_timesteps = list_timesteps ,
                         start = start_time,
                         end = end_time)

# ###

t5b=time.perf_counter()
# ###


distances_clean=np.where(distances_clean==0,np.nan,distances_clean)
## Creation a a sequence matrix
matrice_seq = from_distances_to_sequences_stack(distances_clean)

##Computing relevent number from the sequence matrix

number_of_interaction_sequences = from_stack_to_number_of_interaction_sequences(matrice_seq)
number_of_daily_interaction = from_seq_to_daily_interactions(matrice_seq,list_timesteps)
average_duration_of_an_interaction = from_seq_to_average_interaction_time(matrice_seq)

t6=time.perf_counter()

# #####################
# #Visualisation#
# #####################

# Heatmap for the number of intercations couple wise
vi.heatmap_interactions_number(number_of_interaction_sequences,list_id)


## PLotting the double boxplot for each cows : number of interaction and avearge time:

vi.barplot_interaction_cows(number_of_daily_interaction,average_duration_of_an_interaction ,list_id)

## All around boxplot
vi.boxplot_average_time_number_interactions(number_of_daily_interaction, average_duration_of_an_interaction)


t7=time.perf_counter()
