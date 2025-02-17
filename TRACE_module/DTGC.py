#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np 
import pandas as pd 





def create_file_x_y_t(stack: np.ndarray,
                      list_timesteps:list[str]
                      
                      ):
    """
    
    
    """
    
    serie_timestamp=pd.to_datetime(pd.Series(list_timesteps)).astype("int64")
    normalized_tstamp=(serie_timestamp - serie_timestamp.min())/(serie_timestamp.max()-serie_timestamp.min())
    
    
    txt=""
    coord_triu=np.triu_indices(stack[0,:,:].shape[0],k=1)
    coord_triu=[(coord_triu[0][i],coord_triu[1][i]) for i in range(coord_triu[0].shape[0])]
   
    for idx,val in normalized_tstamp.items() : 
        
        coordonnees = np.argwhere(stack[idx,:,:]==1)
        
        for couple in coordonnees: 
           
            if tuple(couple) in coord_triu: ## Pour n'avoir qu'une seule arrête, à enlever si on veut les arrêtes dans les deux sens
            
                txt+="{} {} {} \n".format(*couple,val)
        
    
  
    
    return txt




