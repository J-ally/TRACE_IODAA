# Module d'implémentation du papier https://doi.org/10.48550/arXiv.1612.09259

from datetime import datetime,timedelta
from tqdm import tqdm
from utils import *

###  Définition des types 
type Individu = str  # Accelero_id de la vache
type TimeStep = datetime      # Timestep en seconde ?
type Sequence = list[Interaction]  # Séquence d'intéraction
type ListeMotif = list[Motif]  # Liste de motifs


class Arc :
    def __init__(self, ind1 : Individu, ind2 : Individu, oriented : bool = True):

        self._ind1 = ind1 
        self._ind2 = ind2 
        self._oriented = oriented
    
    def __repr__(self):
        if self._oriented :
            sep = "->"
        else :
            sep = "--"
        return(f"{self._ind1} {sep} {self._ind2}")
    
    def __eq__(self, obj) : 
        if self._oriented :
            result = ((self._ind1 == obj._ind1) & (self._ind2 == obj._ind2))
        else :
            result = (((self._ind1 == obj._ind1) & (self._ind2 == obj._ind2)) 
                      | ((self._ind1 == obj._ind2) & (self._ind2 == obj._ind1)))
        return result

    def __hash__(self):
        if self._oriented :
            hashed = hash((self.ind1, self.ind2))
        else :
            hashed = hash(frozenset((self.ind1, self.ind2)))   
        return hashed

    @property
    def inds(self):
        return (self._ind1, self._ind2)

    @property
    def ind1(self): 
        return self._ind1

    @property
    def ind2(self): 
        return self._ind2
    
    @property
    def oriented(self):
        return self._oriented
    
    @oriented.setter
    def oriented(self,val):
        if not isinstance(val, bool): 
            raise ValueError("l'attribut oriented doit être un booléen")
        else :
            self._oriented = val
    
    


class Interaction :
    def __init__(self, ind1 : Individu, ind2 : Individu, timestep : datetime) : 
        """Interaction entre l'individu 1 et 2 à un timestep donné
        L'individu 1 capte l'individu 2 à un instant donné

        Args:
            ind1 (Individu): Identifiant de l'individu 1
            ind2 (Individu): Identifiant de l'individu 2
            timestep (datetime): Date de l'interaction
        """
        self._ind1 = ind1
        self._ind2 = ind2
        self._ts = timestep
    
    def __repr__(self): 
        return f" {self.ind1} -> {self.ind2} ({self.ts})"

    def __lt__(self,obj):
        return (self._ts < obj._ts)
    
    def __gt__(self,obj):
        return (self._ts > obj._ts)
    
    def __eq__(self,obj):
        return ((self._ind1 == obj._ind1) & (self._ind2 == obj._ind2) & (self._ts == obj._ts))
    
    @property
    def inds(self):
        return (self._ind1, self._ind2)
    
    @property
    def ts(self): 
        return self._ts

    @property
    def ind1(self): 
        return self._ind1

    @property
    def ind2(self): 
        return self._ind2


    
class Motif:
    def __init__(self,*args : Arc, oriented : bool = True):
        """ Objet Motif : séquence d'arc ordonnés)
        """
        self._oriented = oriented
        self.list_arc = list(args)
        for arc in self.list_arc : arc._oriented = self._oriented

        self.dict_arc = dict(zip([i for i in range(len(self.list_arc))], self.list_arc))
        

    def __repr__(self):
        to_print = ""
        for key,arc in self.dict_arc.items() : 
            to_print += f"{key} : {arc} \n "
        to_print = f"Motif({to_print}, Oriented : {self._oriented})"
        return to_print
    
    def __getitem__(self,key):
        return self.dict_arc[key]
    
    def __eq__(self,motif):
        return (set(self.list_arc) == set(motif.list_arc))
    
    def __hash__(self):
        return hash(frozenset(self.list_arc))

    def __len__(self): 
        return len(self.list_arc)
    
    @property
    def oriented(self): 
        return self._oriented
    
    @oriented.setter
    def oriented(self,val):
        if not isinstance(val, bool): 
            raise ValueError("l'attribut oriented doit être un booléen")
        else :
            self._oriented = val
            for arc in self.list_arc : arc._oriented = self._oriented
            
    
    def __add__(self, arc : Arc):
        """Ajout d'un arc en fin de séquence de motif

        Args:
            arc (Arc): arc à ajouter

        Returns:
            Motif : Motif avec l'arc ajouté au début 
        """
        list_arc = self.list_arc + [arc]
        return Motif(*list_arc, oriented= self._oriented)
    
    def add_suffix(self, arc : Arc):
        """Ajout d'un arc en début de séquence de motif

        Args:
            arc (Arc): arc à ajouter

        Returns:
            Motif : Motif avec l'arc ajouté au début 
        """
        list_arc = [arc] + self.list_arc
        return Motif(*list_arc, oriented= self._oriented)

    
    def graph(self):
        """
        Représentation graphique du motif
        """
        pass

    def gen_submotif(self):
        """Fonction du génére l'ensemble des sous motifs du motif

        Returns:
            list[Motif] : liste des sous motifs (ne contient pas le motif nul, et contient le motif lui même)
        """
        list_sub_motif = [] #Liste qui va contenir les sous motifs 
        if self._oriented : 
            sub_seq_keys = subset_int_consecutif(list(self.dict_arc.keys())) # On récupère les clées de tous les sous motifs
        else :
            sub_seq_keys = subset_int(list(self.dict_arc.keys()))
        for list_keys in sub_seq_keys : 
            list_sub_motif.append(Motif(*[self.dict_arc[key] for key in list_keys], oriented= self._oriented)) # Instanciation des motifs à partir des arc
        return list_sub_motif
        
        
    
    
    

def count_instance_motif(sequence: Sequence, motif: Motif, delta: int) -> int:
    """
    Fonction qui permet de calculer le nombre d'occurences du motif `motif` dans la séquence d'intéraction `sequence` dans une fenetre de temps `delta`

    Parameters:
    ________________
    - sequence (Sequence) : Séquence d'intéraction ordonnée par pas de temps
    - motif (Motif) : Motif d'intérêt à détecter
    - delta (int) : Durée (en s) de la fenêtre de temps dans laquelle on veut considérer le motif 

    Returns:
    ________________
    - int : Nombre d'occurence du motif `motif` dans la séquence d'intéraction

    """
    # On tri la séquence d'intéraction 
    sequence.sort()
    delta = timedelta(seconds = delta)

    # On récupère la longueur du motif :
    l_motif = len(motif)
    # Génération de tous les sous motifs 
    submotifs = motif.gen_submotif()
    l_submotifs = len(submotifs) #Nombre de sous-motifs 

    # Dictionnaire de comptage qui a pour clé un motif (cf figure 2 du papier)
    counts = dict(zip(submotifs, [0 for i in range(l_submotifs)]))
    dict_counts = dict(zip(submotifs, [[0] for i in range(l_submotifs)]))
    start = 0 

    # Début de la boucle de counts 
    for end in tqdm(range(len(sequence))):
        while sequence[start].ts + delta < sequence[end].ts :
            # Decrement Counts 
            counts[Motif(sequence[start].inds, oriented= motif._oriented)] -= 1
            for suffix in counts.keys() :
                # Si le motif est trop grand
                if len(suffix) >= l_motif - 1 : 
                    continue
                else :
                    concat = suffix.add_suffix(sequence[start].inds)
                    if concat not in counts.keys() : continue #Motif n'est pas dans les sous motifs que l'on recherche
                    else :
                        counts[concat] -= counts[suffix] #Motif connu
            start +=1

        # Increment counts
        for prefix in reversed(list(counts.keys())) : 
            if len(prefix) >= l_motif : continue
            else : 
                concat = prefix + sequence[end].inds
                if concat not in counts.keys() : continue #Motif n'est pas dans les sous motifs que l'on recherche
                else : counts[concat] += counts[prefix]
        counts[Motif(sequence[end].inds, oriented= motif._oriented)] += 1
        
        # Ajout des données de counts dans le dictionnaire
        for motif in submotifs : 
            dict_counts[motif].append(counts[motif])

    return counts, dict_counts
    
        
            
    
    




