#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:25:58 2025

@author: bouchet
"""
from motif import Interaction
from datetime import datetime
import numpy as np 
import tqdm

def get_list_interactions(
       
        stack : np.ndarray, 
        list_id : list[str],
        list_time_steps : list[datetime]
        ) -> list[Interaction]: 
        
    list_sequence=[]
        
    for i in tqdm.tqdm(list_id): 
        for j in list_id[list_id.index(i):]: 
            
            index_i,index_j=list_id.index(i),list_id.index(j)
            
            for t in range(len(list_time_steps)) : 
               
                if stack[t,index_i,index_j]==1 : 
                    
                    list_sequence.append(Interaction(i,j,list_time_steps[t]))
                    
        
    return list_sequence


