#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:18:59 2025

@author: bouchet
"""
import pandas as pd 
import numpy as np 



def create_test_data(
        thresh : float 
        ) -> tuple : 
            
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
           
            matrice_symetrie[i,:,:]=np.fmax(matrice_RSSI[i,:,:], matrice_RSSI[i,:,:].T)
            matrice_symetrie_adjacence[i,:,:]=np.where(matrice_symetrie[i,:,:]>thresh,1,0)
            
            
            
            
            
                                 
        df_test=pd.DataFrame(data)
        return df_test, matrice_RSSI,matrice_symetrie,matrice_symetrie_adjacence