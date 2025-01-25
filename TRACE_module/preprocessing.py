#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 23:03:45 2025

@author: bouchet
"""


import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
from multiprocessing import Pool, cpu_count
from itertools import combinations
from mlxtend.frequent_patterns import apriori
import networkx as nx

def lissage_signal(
    smooth_time: str = "20s",
    smooth_function: str = "mean",
    DataFrame: pd.DataFrame = None,
) -> pd.DataFrame:
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

    file = DataFrame.copy()
    file.set_index("glob_sensor_DateTime", inplace=True)
    smoothed_file = (
        file.groupby("accelero_id")
        .resample(smooth_time)
        .agg({"RSSI": smooth_function})
        .reset_index()
    )
    smoothed_file.set_index("glob_sensor_DateTime", inplace=True)
    smoothed_file.dropna(axis=0, inplace=True, subset=["RSSI"])
    smoothed_file.reset_index(inplace=True)

    return smoothed_file




def format_df_to_concatenation(file: str, smooth_time: str = "20s") -> pd.DataFrame:
    """

    Cette fonction prend en argument un CHEMIN de fichier (traité après lissage).
    Le nom du fichier doit impérativement correspondre au format d'origine des fichiers ( doit être accessible par .split("_")[1] )
    La fonction retourne un dataframe, auquel on a ajouté la colonne id_sensor.
    Ainsi, la fonction :
        - Ouvre les fichiers parquts à partir du chemin spécifié
        - le formate (ajout de la colonne id_sensor)
        - renvoie un DataFrame
    """

    id_sensor = os.path.split(file)[1].split("_")[1]

    dataframe = pd.read_parquet(file, engine="pyarrow")
    # print("1", dataframe.columns)
    dataframe = lissage_signal(DataFrame=dataframe, smooth_time=smooth_time)
    dataframe["id_sensor"] = id_sensor

    # print(2, dataframe.columns)
    return dataframe




def concatenate_df(liste_files: list[str], smooth_time: str = "20s") -> pd.DataFrame:
    """
    Cette fonction prend en argument une liste de chemin fichiers parquets, les concatène, et retourne un DataFrame entier
    Il serait bon de vérifier en ammont que la liste ne contient que les fichiers souhaités ?
    """

    concatenated_df = pd.DataFrame()
    for file in liste_files:
        concatenated_df = pd.concat(
            [concatenated_df, format_df_to_concatenation(file, smooth_time=smooth_time)])

    return concatenated_df




def transform_rssi_to_distance(
    dataframe: pd.DataFrame, type_evaluation: str = "log"
) -> pd.DataFrame:
    """
    Cette fonction permet de convertir le signal en une mesure de distance.
    Elle applique une transformation sur la colonne 'RSSI' et l'applique à une nouvelle colonne créee, "evaluated_distance"
    Elle renvoie un dataframe

    On pourra enrirchir cette fonction à travers l'ajout de modalités "type_evaluation" (défaut : log)

    type_evaluation : log,

    Modèle "log" : On utilise le modèle RSSI = P0 - 10nlog10(d) + X. On considère P0,X=0 (distances définies à une constante près, et n égal à 3.
    Il est important de noter qu'avec cette méthode TRES simpliste, les distances ne sont pas interprétables en mêtres

    """

    if type_evaluation == "log":
        dataframe["evaluated_distance"] = 10 ** (-dataframe["RSSI"] / (10 * 3))

        return dataframe

def create_stack(
        dataframe : pd.DataFrame, 
        list_id : list[str], 
        distance_eval: str = "RSSI", 
        symetrisation : bool=True, 
        adjacence : bool=True, 
        threshold : float=-65., 
        
        )  -> tuple((np.array, list[str])) :
        

        """
        
        
        """
        
        id_to_index = {id_: idx for idx, id_ in enumerate(list_id)}

       
        dataframe["row_idx"] = dataframe["accelero_id"].map(id_to_index)
        dataframe["col_idx"] = dataframe["id_sensor"].map(id_to_index)

       
        nb_id = len(list_id)
        list_timesteps = pd.unique(dataframe["glob_sensor_DateTime"])
        
        

        
        timestep_to_idx = {timestep: idx for idx, timestep in enumerate(list_timesteps)}

       
        timestep_indices = dataframe["glob_sensor_DateTime"].map(timestep_to_idx)
        row_indices = dataframe["row_idx"].to_numpy()
        col_indices = dataframe["col_idx"].to_numpy()
        values = dataframe[distance_eval].to_numpy()



        stack = np.full((len(list_timesteps), nb_id, nb_id), np.nan)
        stack[timestep_indices, row_indices, col_indices] = values

        if symetrisation : 
            if distance_eval in ["RSSI"] : 
                 stack=np.fmax(stack,stack.transpose(0,2,1)) 
            else :
                stack=np.fmin(stack,stack.transpose(0,2,1)) 
        if adjacence: 
            if distance_eval in ["RSSI", "evaluated_distance"] : 
                
                stack=np.where(stack > threshold, 1,0 )
                
        return stack,list_timesteps           
   
    
   
    
def crop_start_end_stack(stack  : np.ndarray, 
                         list_timesteps : pd.Series , 
                         start : pd.Timestamp, 
                         end : pd.Timestamp
                        ) -> tuple:
        """
    
    

    Parameters
    ----------
    stack : np.ndarray
        DESCRIPTION.
    list_timesteps : pd.Series
        DESCRIPTION.
    start : pd.Timestamp
        DESCRIPTION.
    end : pd.Timestamp
        DESCRIPTION.

    Returns
    -------
    tuple
        DESCRIPTION.

    """
    
        time_filter = (list_timesteps >= start) & (list_timesteps <= end)
        filtered_timestamps = list_timesteps[time_filter]
        stack_cropped = stack[time_filter]
        
        
        return stack_cropped, filtered_timestamps
    
    
    
def stack_to_one_hot_df(stack : np.ndarray, 
                        list_id : list[str]) -> pd.DataFrame: 
    
    """
    
    
    """
    
    couples = ["_".join(couple) for couple in combinations(list_id, 2)]
    df=pd.DataFrame(columns=couples)
    
    
    
    
    for c in couples: 
        idx=(list_id.index(c.split("_")[0]),list_id.index(c.split("_")[1]))
   
        df[c]=stack[:,*idx]
        
    return df.astype(bool)



def apriori_(df : pd.DataFrame,
             min_support : float,
             min_number : int) -> pd.DataFrame:
    
    motifs=apriori(df,min_support=min_support,use_colnames=True)
    filtered_motifs=motifs[motifs['itemsets'].apply(len)>=min_number].reset_index()
    
    filtered_motifs.sort_values(by="support",ascending=False,inplace=True)
                                
    return filtered_motifs



def apply_sousgraph_connexe_maximum(x : frozenset) -> frozenset: 
    
    
    """
    renvoie les sommets de la plus grande composante connexe du motif
    
    """
    
    graph=nx.Graph() 
    
    for edge in x: 
        n1,n2=edge.split("_")
        graph.add_edge(n1,n2)
    connected_components = nx.connected_components(graph)
    largest_component_nodes = max(connected_components, key=len)
    
    return frozenset(largest_component_nodes)

def get_maximum_connex_graph(dataframe : pd.DataFrame) -> pd.DataFrame:
    
    """
    
    
    """
    
    dataframe["motif_connexe_maximum"]=dataframe["itemsets"].apply(apply_sousgraph_connexe_maximum)
        
    return dataframe
        
    
    
    
    
    
        
        


                    
    




            
