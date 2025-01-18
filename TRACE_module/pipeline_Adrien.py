
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

import numpy as np
import os 
import pandas as pd

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


stack,l=pp.create_stack(data,list_id)
t5=time.perf_counter()
print("étape 1 : {} \n étape 2 : {} \n étape 3 {} \n étape 4 : {}".format(t2-t1,t3-t2,t4-t3,t5-t4))