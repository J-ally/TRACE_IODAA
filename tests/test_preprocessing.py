#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 23:30:59 2025

@author: bouchet
"""

import pytest
import pandas as pd

import numpy as np
import sys

sys.path.insert(1, '/Users/bouchet/Documents/Cours/Cours /AgroParisTech /3A/IODAA/PFR/TRACE_IODAA')


from UTILS_module.create_data import create_test_data
from TRACE_module.preprocessing import create_stack,lissage_signal



        
    
@pytest.mark.parametrize("thresh", [-45,-70,-80])
def test_creation_stack(thresh) : 
    
    df,m1,m2,m3,=create_test_data(thresh) 
    list_id=["a","b","c","d","e"]
    assert np.array_equal(np.nan_to_num(m1, nan=1000), np.nan_to_num(create_stack(df, list_id, symetrisation=False, adjacence=False,threshold=thresh)[0], nan=1000)), "Les matrices m1 et le résultat de create_stack (symetrisation=False, adjacence=False) ne sont pas égales."
    assert np.array_equal(np.nan_to_num(m2, nan=1000), np.nan_to_num(create_stack(df, list_id, symetrisation=True, adjacence=False,threshold=thresh)[0], nan=1000)), "Les matrices m2 et le résultat de create_stack (symetrisation=True, adjacence=False) ne sont pas égales."
    assert np.array_equal(np.nan_to_num(m3, nan=1000), np.nan_to_num(create_stack(df, list_id, symetrisation=True, adjacence=True,threshold=thresh)[0], nan=1000)), "Les matrices m1 et le résultat de create_stack (symetrisation=True, adjacence=True) ne sont pas égales."
    
    
    


def test_lissage_signal():
  
    date_depart = pd.Timestamp("2024-01-01 00:00:00")
    duree_simulation = 24 * 60 * 60  # 24 heures
    periode_pattern = 20
    pattern_rssi = {5: -40, 10: -55, 15: -70, 0: -80}  # 0 correspond à la 20e seconde
    timestamps = pd.date_range(start=date_depart, periods=duree_simulation, freq="1S")
    intensites = [
        pattern_rssi.get(i % periode_pattern, np.nan) for i in range(1, duree_simulation + 1)
    ]

    captants = ['capteur_A' for _ in range(len(timestamps))]
    captes = ['capteur_B' for _ in range(len(timestamps))]

    df = pd.DataFrame({
        "glob_sensor_DateTime": timestamps,
        "accelero_id": captants,
        "id_sensor": captes,
        "RSSI": intensites
    })
    
    df=lissage_signal(DataFrame=df)

    ################################################
    ### Création d'un dataframe lissé 20s, par mean
    ################################################
    timestamps2 = pd.date_range(start=date_depart, periods=duree_simulation // 20, freq="20S")

    signal = [-61.25 for _ in range(len(timestamps2))]


    df_signal_20s = pd.DataFrame({
        "glob_sensor_DateTime": timestamps2,
        "RSSI": signal
    })
    
    pd.testing.assert_series_equal(df["RSSI"], df_signal_20s["RSSI"], check_index_type=False,check_exact=True, check_dtype=True)



 





