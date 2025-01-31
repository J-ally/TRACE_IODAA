#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from plotnine import ggplot, geom_line, aes, scale_x_discrete


def date_to_index(date: str, list_date: pd.Series):
    return list_date[list_date == date].index[0]


def visualisation_distance_matrice(
    matrice: np.ndarray, labels: list[str], max_val: int, min_val: int = 0
) -> None:
    # Créer un masque pour les NaN
    mask = np.isnan(matrice)

    # Palette colorblind-friendly (cividis est perceptuellement uniforme)

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        matrice,
        annot=True,
        cmap="GnBu_r",  ## Pour Timothée qui est daltonien
        fmt=".2f",
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        linecolor="black",
        linewidths=1,
        mask=mask,
        vmin=min_val,  # Échelle min/max pour inclure toutes les valeurs non-NaN
        vmax=max_val,
    )

    # Inversion de l'échelle (palette claire pour valeurs élevées)
    plt.gca().invert_yaxis()  # Optionnelle : inverser l'axe Y pour alignement conventionnel

    # Ajout de la couleur grise pour les NaN
    plt.gca().set_facecolor("white")  # Couleur des cellules NaN

    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(rotation=45, fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_accelero_ranking(df: pd.DataFrame):

    """
    Fonction permettant d'afficher sous la forme d'un histogramme du classement des individus proches par animal

    **Args :
        df (Dataframe) : Dataframe contenant au moins les attributs "id_sensor" et "accelero_id" et un attribut de comptage (par exemple "count")

    **Returns : Histogramme
    """

    # Créer un graphique en barres groupées
    plt.figure(figsize=(10, 6))

    # Tracer les barres avec Seaborn
    sns.barplot(
        data=df,
        x='id_sensor',
        y='count',
        hue='accelero_id',  # Couleurs différentes pour chaque accelero_id
        dodge=True  # Pour séparer les barres par groupe
    )

    # Ajouter des labels et un titre
    plt.xlabel('ID Sensor')
    plt.ylabel('Nombre d\'occurrences')
    plt.title('Classement des individus proches par animal')
    plt.legend(title='Accelero ID')

    # Afficher le graphique
    plt.show()

def heatmap_interactions_number(
        matrix_interaction_number : np.ndarray,
        list_id : list[str]
        ) -> None :

    """
    Function that traces a heatmap with the number of interactio for each cow couple

    Parameters:
    -----------
    matrix_interaction_number : np.ndarray
        2D array with the number of daily interactions for each cow couple
    list_id : list[str]
        List with the cows ID
    --------

    """

    ## Plot the heatmap of the interactions number for each couple
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix_interaction_number , cmap='hot', interpolation='nearest')
    plt.colorbar(label="Maximum Sequence ID")
    plt.title(" Between Cows")
    plt.xlabel("Cow ID (Column)")
    plt.ylabel("Cow ID (Row)")
    plt.xticks(ticks=np.arange(len(list_id)), labels=list_id, rotation=45)
    plt.yticks(ticks=np.arange(len(list_id)), labels=list_id)
    plt.show()


def barplot_interaction_cows(
        number_of_daily_interaction : np.ndarray,
        average_duration_of_an_interaction : np.ndarray,
        list_id : list[str]
        ) -> None:
    """
    Function that traces 2 boxplots for each cows : number of daily interaction and the duration of the interactions

    Parameters:
    -----------
    number_of_daily_interaction : np.ndarray
        1D array with the number of daily interactions for each cow.
    average_duration_of_an_interaction : np.ndarray
        1D array with an average interaction time for each cow
    list_id : list[str]
        List with the cows ID
    --------

    """

    fig, ax = plt.subplots(figsize=(12, 6))
    # Bar width and x-axis positions
    bar_width = 0.35
    x = np.arange(1, 17)  # Cow indices (1 to 16)

    # Plotting the bars
    bars1 = ax.bar(x - bar_width / 2, number_of_daily_interaction, bar_width, label='Avg Daily Interactions')
    bars2 = ax.bar(x + bar_width / 2, average_duration_of_an_interaction, bar_width, label='Avg Interaction Time (mins)')

    # Labeling the chart
    ax.set_xlabel('Cow Index')
    ax.set_ylabel('Values')
    ax.set_title('Comparison of Average Daily Interactions and Average Interaction Time for Each Cow')
    ax.set_xticks(x)
    ax.set_xticklabels(list_id)
    ax.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()

def boxplot_average_time_number_interactions(number_of_daily_interaction : np.array,
                                             average_duration_of_an_interaction : np.array
                                             ) -> None :

    """
    Function that traces 2 boxplots : number of daily interaction and the duration of the interactions

    Parameters:
    -----------
    number_of_daily_interaction : np.ndarray
        1D array with the number of daily interactions for each cow.
    average_duration_of_an_interaction : np.ndarray
        1D array with an average interaction time for each cow
    --------

    """

    # Prepare box plot data
    box_data_daily_interactions = number_of_daily_interaction
    box_data_mean_duration = average_duration_of_an_interaction

    # Plotting box plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Box plot for daily interactions
    axes[0].boxplot(box_data_daily_interactions, vert=True, patch_artist=True)
    axes[0].set_title("Daily Interaction Counts per Cow")
    axes[0].set_xlabel("Cows")
    axes[0].set_ylabel("Mean Daily Interactions")

    # Box plot for mean interaction duration
    axes[1].boxplot(box_data_mean_duration, vert=True, patch_artist=True)
    axes[1].set_title("Mean Interaction Duration per Cow")
    axes[1].set_xlabel("Cows")
    axes[1].set_ylabel("Mean Interaction Duration (Hours)")

    plt.tight_layout()
    plt.show()



def proba_interaction_motif_along_time( motif : frozenset, 
                                       stack : np.ndarray, 
                                       list_id : list[str], 
                                       list_timesteps
                                       ) -> None: 
    
    """
    
    
    
    """
    
    interaction_vector=np.full(stack.shape[0],0)
    
    masque=np.full(stack.shape[1:3], 0)
    for i in motif : 
        masque[list_id.index(i.split("_")[0]),list_id.index(i.split("_")[1])]=1
      
    bool_vecteur = []
    for i in range(stack.shape[0]):
        bool_vecteur.append(np.array_equal(stack[i] & masque, masque)) 
   
    bool_vecteur = np.array(bool_vecteur)
    
        
    df = pd.DataFrame({
        'timestamp': list_timesteps,
        'is_true': bool_vecteur
    })
    
    # Extraire heure::minute comme nouvelle colonne
    df['time'] = df['timestamp'].dt.strftime('%H:%M')
    
    # Calculer la moyenne de `is_true` pour chaque heure
    result = df.groupby('time')['is_true'].mean().reset_index()
    
    # Renommer les colonnes pour correspondre au résultat attendu
    result.columns = ['time', 'probability']
    result["time"]=result["time"].astype('<M8[ns]')
    plot=(
    ggplot(result,aes(x="time",y="probability")) + geom_line() 
    
    #+ scale_x_discrete(breaks=range(0, 24, 2))
    )
    print(plot)
    return result
    

def proba_interaction_motif_along_time_2( motif : frozenset, 
                                       stack : np.ndarray, 
                                       list_id : list[str], 
                                       list_timesteps
                                       ) -> None: 
    
    """
    
    
    
    """
    
    interaction_vector=np.full(stack.shape[0],0)
    
    masque=np.full(stack.shape[1:3], 0)
    for i in motif : 
        masque[list_id.index(i.split("_")[0]),list_id.index(i.split("_")[1])]=1
      
    bool_vecteur = []
    for i in range(stack.shape[0]):
        bool_vecteur.append(np.array_equal(stack[i] & masque, masque)) 
   
    bool_vecteur = np.array(bool_vecteur)
    
        
    df = pd.DataFrame({
        'timestamp': list_timesteps,
        'is_true': bool_vecteur
    })
    
    # Extraire heure::minute comme nouvelle colonne
    df['time'] = df['timestamp'].dt.strftime('%H:%M')
    df['day']=df['timestamp'].dt.strftime('%d')
    
    
    
    # Renommer les colonnes pour correspondre au résultat attendu
   
    df["time"]=df["time"].astype('<M8[ns]')
    #df["day"]=df["day"].astype('<M8[ns]')
    plot=(
    ggplot(df,aes(x="time",y="is_true",color="day")) + geom_line() 
    
    #+ scale_x_discrete(breaks=range(0, 24, 2))
    )
    print(plot)
    