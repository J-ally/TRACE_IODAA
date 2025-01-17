#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 23:30:59 2025

@author: bouchet
"""

import pytest
import pandas as pd
import numpy as np


import preprocessing as pp
import visualisation as vi 
import motif_2_adrien as motif2
import numpy as np
import os 
import pandas as pd
import dotenv
import sys
import time
from env_loading import parent_dir, output_dir
import preprocessing_2 as pp2
import sys
import os


def create_test_data() : 
            
        start_date = "2024-01-01 00:00:00"
        end_date = "2024-01-15 23:59:59"
        time_steps = pd.date_range(start=start_date, end=end_date, freq="20S")
        data = {
            "glob_sensor_DateTime": [],
            "accelero_id": [],
            "id_sensor": [],
            "RSSI": [],
        }
        list_id=["a","b","c","d","e"]
        
        matrice_RSSI=np.full((len(time_steps),len(list_id),len(list_id)),np.nan)
        matrice_symetrie=np.full((len(time_steps),len(list_id),len(list_id)),np.nan)
        matrice_symetrie_adjacence = np.full((len(time_steps),len(list_id),len(list_id)),np.nan)
        
       
        for i, timestep in enumerate(time_steps):
            
            
         
            
            
            # a capte b avec -70
            data["glob_sensor_DateTime"].append(timestep)
            data["accelero_id"].append("a")
            data["id_sensor"].append("b")
            data["RSSI"].append(-70)
            
            # b capte a avec -60
            data["glob_sensor_DateTime"].append(timestep)
            data["accelero_id"].append("b")
            data["id_sensor"].append("a")
            data["RSSI"].append(-60)
             
            # a et e se captent mutuellement avec -50
            data["glob_sensor_DateTime"].append(timestep)
            data["accelero_id"].append("a")
            data["id_sensor"].append("e")
            data["RSSI"].append(-50)
        
            data["glob_sensor_DateTime"].append(timestep)
            data["accelero_id"].append("e")
            data["id_sensor"].append("a")
            data["RSSI"].append(-50)
            
            
            # Toutes les 40 secondes, c capte d avec -70
            if i % 2 == 0:
                matrice_RSSI[i,:,:]=np.array([[np.nan,-70,np.nan,np.nan,-50],
                                     [-60,np.nan,np.nan,np.nan,np.nan],
                                     [np.nan,np.nan,np.nan,-70,np.nan],
                                     [np.nan,np.nan,np.nan,np.nan,np.nan],
                                     [-50,np.nan,np.nan,np.nan,np.nan],
                                    ])
                
                data["glob_sensor_DateTime"].append(timestep)
                data["accelero_id"].append("c")
                data["id_sensor"].append("d")
                data["RSSI"].append(-70)
            else :
                matrice_RSSI[i,:,:]=np.array([[np.nan,-70,np.nan,np.nan,-50],
                                     [-60,np.nan,np.nan,np.nan,np.nan],
                                     [np.nan,np.nan,np.nan,np.nan,np.nan],
                                     [np.nan,np.nan,np.nan,np.nan,np.nan],
                                     [-50,np.nan,np.nan,np.nan,np.nan],
                                    ])
           
            matrice_symetrie[i,:,:]=np.fmin(matrice_RSSI[i,:,:], matrice_RSSI[i,:,:].T)
            matrice_symetrie_adjacence[i,:,:]=np.where(matrice_symetrie[i,:,:]<-65,1,0)
            
            
            
            
            
                                 
        df_test=pd.DataFrame(data)
        return df_test, matrice_RSSI,matrice_symetrie,matrice_symetrie_adjacence
        
    

def test_creation_stack() : 
        
    df,m1,m2,m3=create_test_data() 
   #return df,m1
    list_id=["a","b","c","d","e"]
    # assert np.array_equal(m1,pp2.create_stack(df, list_id,symetrisation=False, adjacence=False)[0])
    # assert np.array_equal(m2,pp2.create_stack(df, list_id,symetrisation=True, adjacence=False)[0])   
    # assert np.array_equal(m1,pp2.create_stack(df, list_id,symetrisation=True, adjacence=True)[0])
    m1b=pp2.create_stack(df, list_id, symetrisation=False, adjacence=False)[0]

   

    assert np.array_equal(np.nan_to_num(m1, nan=1000), np.nan_to_num(m1b, nan=1000)), "Les matrices m1 et le résultat de create_stack (symetrisation=False, adjacence=False) ne sont pas égales."

    assert np.array_equal(np.nan_to_num(m2, nan=1000), np.nan_to_num(pp2.create_stack(df, list_id, symetrisation=True, adjacence=False)[0], nan=1000)), "Les matrices m2 et le résultat de create_stack (symetrisation=True, adjacence=False) ne sont pas égales."

    assert np.array_equal(np.nan_to_num(m3, nan=1000), np.nan_to_num(pp2.create_stack(df, list_id, symetrisation=True, adjacence=True)[0], nan=1000)), "Les matrices m1 et le résultat de create_stack (symetrisation=True, adjacence=True) ne sont pas égales."
    print("tous les tests passés avec succès !! ")