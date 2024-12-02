#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:01:10 2024

@author: bouchet
"""

import pandas as pd 
import os
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import tqdm 



def lissage_signal(smooth_time : str ='20s', smooth_function : str = "mean", DataFrame : pd.DataFrame =None) -> pd.DataFrame:

    """
    Smooth time series signal according to "smooth_time" parameter chosen

    ** Args : 
        path (str) : the path to the parquet file
        smooth_time (str) : Time step chosen to smooth time series
    ** Returns:
        pd.DataFrame : the dataframe smoothed

    ** Additional information : About "smooth_time" argument :
    - Seconds = "s" Ex : 1 second --> 1s
    - Minutes = "min"
    - Hours = "h"
    - Days = "d"
    - Months = "m"

    """


    file=DataFrame.copy()
    file.set_index('glob_sensor_DateTime', inplace = True)
    smoothed_file = file.groupby("accelero_id").resample(smooth_time).agg({"RSSI":smooth_function}).reset_index() 
    smoothed_file.set_index('glob_sensor_DateTime', inplace = True) 
    smoothed_file.dropna(axis=0,inplace=True,subset=["RSSI"])
    smoothed_file.reset_index(inplace=True) 
    
    
    return smoothed_file


def format_df_to_concatenation(file : str ,smooth_time : str="20s") -> pd.DataFrame : 

    """

    Cette fonction prend en argument un CHEMIN de fichier (traité après lissage). 
    Le nom du fichier doit impérativement correspondre au format d'origine des fichiers ( doit être accessible par .split("_")[1] ) 
    La fonction retourne un dataframe, auquel on a ajouté la colonne id_sensor.
    Ainsi, la fonction : 
        - Ouvre les fichiers parquts à partir du chemin spécifié
        - le formate (ajout de la colonne id_sensor)
        - renvoie un DataFrame 
    """
   
    id_sensor=os.path.split(file)[1].split("_")[1]

    dataframe=pd.read_parquet(file,engine="pyarrow")
    print("1",dataframe.columns)
    dataframe=lissage_signal(DataFrame=dataframe,smooth_time=smooth_time)
    dataframe["id_sensor"]=id_sensor 

    print(2,dataframe.columns)
    return dataframe


def concatenate_df(liste_files : list[str] ) -> pd.DataFrame : 


    """
    Cette fonction prend en argument une liste de chemin fichiers parquets, les concatène, et retourne un DataFrame entier
    Il serait bon de vérifier en ammont que la liste ne contient que les fichiers souhaités ?
    """

    concatenated_df=pd.DataFrame() 
    for file in liste_files : 
            concatenated_df=pd.concat([concatenated_df, format_df_to_concatenation(file)])

    return concatenated_df



def transform_rssi_to_distance(dataframe : pd.DataFrame , type_evaluation : str ="log") -> pd.DataFrame : 

    """ 
    Cette fonction permet de convertir le signal en une mesure de distance.
    Elle applique une transformation sur la colonne 'RSSI' et l'applique à une nouvelle colonne créee, "evaluated_distance"
    Elle renvoie un dataframe

    On pourra enrirchir cette fonction à travers l'ajout de modalités "type_evaluation" (défaut : log)

    type_evaluation : log,

    Modèle "log" : On utilise le modèle RSSI = P0 - 10nlog10(d) + X. On considère P0,X=0 (distances définies à une constante près, et n égal à 3.
    Il est important de noter qu'avec cette méthode TRES simpliste, les distances ne sont pas interprétables en mêtres 

    """

    if type_evaluation=="log" : 

       
        dataframe["evaluated_distance"]=10**(-dataframe['RSSI']/(10*3)) 

        return dataframe



def create_distance_matrix(dataframe : pd.DataFrame ,particular_time_step : str ,list_id : list[str] ,distance_eval : str = "evaluated_distance") -> np.ndarray : 
    """
    'glob_sensor_DateTime', 'accelero_id', 'RSSI', 'id_sensor',
       'evaluated_distance'
    En premier lieu, je stock les matrices de chaque time step dans un dictionnaire (associé à une clé contenant le temps exact). 
    
    
    Particular_time_step : permet de selectionner un unique time step particulier

    """
    subset_dataframe=dataframe[dataframe["glob_sensor_DateTime"]==particular_time_step]
    #dataframe=dataframe.reset_index()  
    nb_id = len(list_id) 
    matrice_df = np.ones((nb_id,nb_id))*np.nan # Liste de nan initialement
    #print(matrice_df.shape  )

    matrice_df = pd.DataFrame(matrice_df, index = list_id, columns=list_id) # j'ajoute l'indexation des lignes et des colonnes avec les ID des vaches

    for index,row in subset_dataframe.iterrows():
        vache_capte = row["accelero_id"]
        vache_captant = row["id_sensor"]
        matrice_df.loc[vache_capte, vache_captant] = row[distance_eval]


    return matrice_df.to_numpy()
    
    
    #print(time_steps, "\n ################")
    
    
def create_matrix_stack(dataframe : pd.DataFrame,list_id : list[str], distance_eval :str ="evaluated_distance") -> tuple[np.ndarray, list[str]]: 

    list_timesteps = pd.unique(dataframe["glob_sensor_DateTime"])
    stack_matrix = np.zeros((len(list_timesteps),len(list_id),len(list_id)))
    for ind,t in tqdm.tqdm(enumerate(list_timesteps)) : 
        mat = create_distance_matrix(dataframe,t,list_id,distance_eval)
        stack_matrix[ind,:,:]= mat
    return stack_matrix,list_timesteps


    
    
    
    


