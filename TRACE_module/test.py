### Pour tester les classes Motif, et Interaction

from motif import *
import numpy as np

date_init = datetime.fromisoformat("2024-12-11T12:00:00")

M = Motif(
    ("a","b"),
    ("a","c"),
    ("c","a")
)

sequence = [
    Interaction("a","b", date_init + timedelta(seconds=25)),
    Interaction("a","c", date_init + timedelta(seconds=17)),
    Interaction("a","c", date_init + timedelta(seconds=28)),
    Interaction("a","c", date_init + timedelta(seconds=30)),
    Interaction("a","c", date_init + timedelta(seconds=35)),
    Interaction("c","a", date_init + timedelta(seconds=15)),
    Interaction("c","a", date_init + timedelta(seconds=32)),
]

submotifs = M.gen_submotif()
l_submotifs = len(submotifs) #Nombre de sous-motifs 

# Dictionnaire de comptage qui a pour clé un motif (cf figure 2 du papier)

