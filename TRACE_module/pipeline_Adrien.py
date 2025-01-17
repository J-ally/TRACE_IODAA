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
import motif_2_adrien as motif2
import numpy as np
import os 
import pandas as pd
import dotenv
import sys

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

folder= parent_dir

list_files=list() 

list_files = get_all_files_within_one_folder(folder, True, extension=".parquet")

folder_savings =  os.sep.join([output_dir,"savings"])
if not  os.path.isdir(folder_savings) : 
    os.makedirs(folder_savings)

data=pp.concatenate_df(list_files)

data=pp.transform_rssi_to_distance(data)
list_id=list(pd.unique(data["accelero_id"]))

# stack,list_timesteps=pp.create_matrix_stack(data,list_id,distance_eval="RSSI")

# np.save(os.path.join(folder_savings,"matrice_test.npy"),stack)
# np.save(os.path.join(folder_savings,"list_timesteps.npy"),list_timesteps)


stack_adj_symetrique,time_steps=pp.create_stack_sym_adj(data,list_id,distance_eval="RSSI")

np.save(os.path.join(folder_savings,"matrice_test_adj_sym.npy"),stack_adj_symetrique)
np.save(os.path.join(folder_savings,"list_timesteps_test_adj.npy"),time_steps)

stack_adj_symetrique=np.load(os.path.join(folder_savings,"matrice_test_adj_sym.npy"))
time_steps=np.load(os.path.join(folder_savings,"list_timesteps_test_adj.npy"))

vi.visualisation_distance_matrice(stack_adj_symetrique[40],list_id,min_val=0,max_val=1)

list_interactions_=motif2.get_list_interactions(stack_adj_symetrique, list_id,time_steps)