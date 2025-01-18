#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 17:04:42 2025

@author: bouchet
"""

import os 
import sys 

sys.path.insert(1, '/Users/bouchet/Documents/Cours/Cours /AgroParisTech /3A/IODAA/PFR/TRACE_IODAA')

from TRACE_module.preprocessing import create_stack
from UTILS_module.create_data import create_test_data
from TRACE_module import motif as m


def test_list_interaction() : 
    list_id=["a","b","c","d","e"]
    
    df=create_test_data(-65)[0]
    stack,time_steps=create_stack(df,list_id)
    
    l1=m.get_list_interactions(stack,list_id,time_steps)
    
    assert type(l1)==list, "le résultat de la fonction n'est pas une liste "
    assert type(l1[0])==m.Interaction, "les éléments de la liste ne sont pas des instances de classe Interacrtion"
    
    
    ##Comparaison de la taille de la liste à celle attendue : attention, celle-ci dépend directement des données générées## 
    if len(time_steps)%2 == 0 : 
        len_expected=1.5*len(time_steps)
        
    else : 
        len_exprected=1.5*len(time_steps+1)-1
    assert len(l1)==len_expected, 'la longueur de la liste ne correspond pas à la longueur attendue' 