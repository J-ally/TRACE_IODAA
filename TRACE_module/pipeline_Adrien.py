import visualisation as vi 
import motif_2_adrien as motif2
import numpy as np
import os 
import pandas as pd
import preprocessing as pp


folder="../data/Data_rssi_glob_sensor_time"

##############
#Préparation
##############
list_files=list() 

for f in os.listdir(folder): 
    if f.endswith(".parquet") : 
        list_files.append(os.path.join(folder,f))
        
folder_savings="../savings"
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
        
