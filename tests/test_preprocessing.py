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
from TRACE_module.preprocessing import create_stack



        
    
@pytest.mark.parametrize("thresh", [-45,-70,-80])
def test_creation_stack(thresh) : 
    
    df,m1,m2,m3,=create_test_data(thresh) 
    list_id=["a","b","c","d","e"]
    assert np.array_equal(np.nan_to_num(m1, nan=1000), np.nan_to_num(create_stack(df, list_id, symetrisation=False, adjacence=False,threshold=thresh)[0], nan=1000)), "Les matrices m1 et le résultat de create_stack (symetrisation=False, adjacence=False) ne sont pas égales."
    assert np.array_equal(np.nan_to_num(m2, nan=1000), np.nan_to_num(create_stack(df, list_id, symetrisation=True, adjacence=False,threshold=thresh)[0], nan=1000)), "Les matrices m2 et le résultat de create_stack (symetrisation=True, adjacence=False) ne sont pas égales."
    assert np.array_equal(np.nan_to_num(m3, nan=1000), np.nan_to_num(create_stack(df, list_id, symetrisation=True, adjacence=True,threshold=thresh)[0], nan=1000)), "Les matrices m1 et le résultat de create_stack (symetrisation=True, adjacence=True) ne sont pas égales."
    
    
    
   

    
    
