
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:51:20 2025

@author: bouchet
"""

###############################################################################
#                                 IMPORTS                                     #
###############################################################################

import TRACE_module.preprocessing as pp

import TRACE_module.visualisation as vi
from TRACE_module.descriptive_analysis import from_stack_to_number_of_interaction_sequences,from_distance_to_sequences_vector, from_distances_to_sequences_stack,from_seq_to_average_interaction_time,from_seq_to_daily_interactions
from TRACE_module.apriori_spade import stack_to_one_hot_df, get_maximum_connex_graph, apriori_
from TRACE_module.DTGC import create_file_x_y_t as create_file_x_y_t
from TRACE_module.DTGC import create_file_nodeId_label as create_file_nodeId_label
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

import dotenv
import sys
import time
from TRACE_module.env_loading import parent_dir, output_dir

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

data=pp.concatenate_df(list_files,smooth_time='20s')
data=pp.remove_captor_(data,["366b"])

# ##########################################
# #ETAPE 3 -Transformation RSSI en distance#
# ##########################################
# t3=time.perf_counter()

data=pp.transform_rssi_to_distance(data)
# t4=time.perf_counter()

# ###################################################################
# #ETAPE 5 - Création d'un stack de matrice d'adjacence symétriques #
# ###################################################################

list_id=list(pd.unique(data["accelero_id"]))


stack,list_timesteps=pp.create_stack(data,list_id)
t5=time.perf_counter()

# #########,##########################################################
# #ETAPE 6- Analyses                                                #
# ###################################################################


#start_time = pd.Timestamp('2024-03-20T08:39:00.000000000')
#end_time = pd.Timestamp('2024-04-10T16:36:00.000000000')

#New timestamps without any bagtime
 #start_time = pd.Timestamp('2024-03-22T08:39:00.000000000')
 #end_time = pd.Timestamp('2024-04-8T16:36:00.000000000')


###BUISSON
start_time = pd.Timestamp('2024-10-18T15:00:00.000000000')
end_time = pd.Timestamp('2024-10-27T06:00:00.000000000')



distances_clean,list_timesteps=pp.crop_start_end_stack(stack=stack,
                         list_timesteps = list_timesteps ,
                         start = start_time,
                         end = end_time)


#distances_clean=stack
# ###

t5b=time.perf_counter()
# ##

#distances_clean=np.where(distances_clean==0,np.nan,distances_clean)


# # ## Creation a a sequence matrix
#matrice_seq = from_distances_to_sequences_stack(distances_clean)

# # ##Computing relevent number from the sequence matrix

# number_of_interaction_sequences = from_stack_to_number_of_interaction_sequences(matrice_seq)
# number_of_daily_interaction = from_seq_to_daily_interactions(matrice_seq,list_timesteps)
# average_duration_of_an_interaction = from_seq_to_average_interaction_time(matrice_seq)

# # t6=time.perf_counter()

# # #####################
# # #Visualisation#
# # #####################

# # # Heatmap for the number of intercations couple wise
# # vi.heatmap_interactions_number(number_of_interaction_sequences,list_id)


# # ## PLotting the double boxplot for each cows : number of interaction and avearge time:

# # vi.barplot_interaction_cows(number_of_daily_interaction,average_duration_of_an_interaction ,list_id)

# # ## All around boxplot
# # vi.boxplot_average_time_number_interactions(number_of_daily_interaction, average_duration_of_an_interaction)
# print("ok")

# t7=time.perf_counter()
# ###################################################################
# #ETAPE 7 - Recherche de motifs / Apriori                          #
# # ###################################################################

# print("!!!!!!!!!")
# print(list_id,distances_clean)
# d=stack_to_one_hot_df(distances_clean, list_id)
# print(d.shape,d)
# print("###########")

# motifs=apriori_(d, 0.001, 3)
# print("éééééééééé")
# # t8=time.perf_counter()

# motifs=get_maximum_connex_graph(motifs)


# t9=time.perf_counter()


# a=vi.proba_interaction_motif_along_time(motifs.iloc[0]["itemsets"],distances_clean,list_id,list_timesteps)
# vi.proba_interaction_motif_along_time(motifs.iloc[1]["itemsets"],distances_clean,list_id,list_timesteps)
# vi.proba_interaction_motif_along_time(motifs.iloc[2]["itemsets"],distances_clean,list_id,list_timesteps)


# a=vi.proba_interaction_motif_along_time_2(motifs.iloc[0]["itemsets"],distances_clean,list_id,list_timesteps)
# vi.proba_interaction_motif_along_time_2(motifs.iloc[1]["itemsets"],distances_clean,list_id,list_timesteps)
# vi.proba_interaction_motif_along_time_2(motifs.iloc[2]["itemsets"],distances_clean,list_id,list_timesteps)
# # ###################################################################
# #Benchmarks                                                       #
# ##################################################################

# #print("--BENCHMArk-- \n ==========\n étape 1 - localisation des données: {} \n étape 2 - concaténation des fichiers et création d'un dataframe unique ': {} \n étape 3 -transformation des signaux RSSI en distance {} \n étape 4 - Mise en forme des données sous forme matricielle: {} \n étape 5 : Analyse descriptive :{} \n étape 6 : Extraction de motifs par Apriori :{} \n étape 7 : extraction composante connexe maximale :{}".format(t2-t1,t3-t2,t4-t3,t5-t4,t6-t5, t8-t7,t9-t8))




# import itertools
# import pandas as pd 
# import numpy as np 
# dict_grid={"smooth_time": [ "20s","40s","60s","1800s"],
#            "smooth_function" : ["mean","max","median"], 
#            "threshold" : [-65,-70,-75]
#            }
    
    
# df_total=pd.DataFrame()


# keys = dict_grid.keys()
# values = dict_grid.values()

# combinations = list(itertools.product(*values))

# # Organiser les combinaisons dans un format lisible
# all_combinations = [dict(zip(keys, combo)) for combo in combinations]



# c=0
# for comb in all_combinations : 
    
#     print(c,comb)
#     smooth_time=comb["smooth_time"]
#     smooth_function=comb[ "smooth_function" ]
#     threshold=comb["threshold"]
    
#     data=pp.concatenate_df(list_files,smooth_time=smooth_time,smooth_function=smooth_function)
#     list_id=list(pd.unique(data["accelero_id"]))
#     stack,list_timesteps=pp.create_stack(data,list_id,threshold=threshold)
    
#     start_time = pd.Timestamp('2024-10-16T15:00:00.000000000')
#     end_time = pd.Timestamp('2024-10-29T06:00:00.000000000')



#     distances_clean,list_timesteps=pp.crop_start_end_stack(stack=stack,
#                              list_timesteps = list_timesteps ,
#                              start = start_time,
#                              end = end_time)

        
        
#     d=pp.stack_to_one_hot_df(distances_clean, list_id)
    
#     motifs=pp.apriori_(d, 0.0015, 3)
    
#     motifs=pp.get_maximum_connex_graph(motifs)

#     motifs["smooth_time"]=smooth_time
#     motifs["smooth_function"]=smooth_function
#     motifs["threshold"]=threshold
    
    
#     df_total=pd.concat([df_total,motifs])
#     c+=1

txt=create_file_x_y_t(distances_clean, list_timesteps)
with open("cows.txt","w") as file : 
    file.write(txt)

txt=create_file_nodeId_label(list_id)

with open("cows_node_id_label.txt","w") as file : 
    file.write(txt)
    
    
    

