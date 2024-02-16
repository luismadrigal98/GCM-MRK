"""
This script contains the definition of the RandPartition class. An individual is represented by a list of integers, which denotes the assignment of that individual
to a given cluster. An individual can be understood, then, as a single partition. Individuals can be initialized randomly, or can be initialized taking as
input partitions generated from gene-coexpression analysis software, like Clust or WGCNA (See class InpPartition).

"""

import random
from davies_bouldin_index import davies_bouldin_index
import pairwise_distances
from bic import bic
from aic import aic

class RandPartition:
    """
    A class to represent a random partition of genes into clusters.
    """

    def __init__(self, element_number, G=2, all_in_clusters=True):
        """
        Initialize the instance variables and create a random partition.
        """

        self.element_number = element_number
        self.__init_part(element_number, G, all_in_clusters)

    def __len__(self):
        """
        Return the length of the chromosome in the individual.
        """

        return len(self.items)
    
    def __init_part(self, element_number, G, all_in_clusters=True):
        """
        Create a random partition of genes into clusters.
        """

        start = 1 if all_in_clusters else 0
        self.items = [random.randint(start, G) for _ in range(element_number)]

    def get_values(self, Data, DBI = True, BIC = True, AIC = True):
        """
        Calculate different internal measurements of cluster quality.
        """
        
        Distance_matrix = pairwise_distances.pairwise_d(Data)

        internal_indexes = {}

        if DBI:
            internal_indexes["DBI"] = davies_bouldin_index(Distance_matrix, self.items, unassigned_penalty=1.0)

        if BIC:
            internal_indexes["BIC"] = bic(Distance_matrix, self.items, unassigned_penalty=1.0)

        if AIC:
            internal_indexes["AIC"] = aic(Distance_matrix, self.items, unassigned_penalty=1.0)

        return tuple(internal_indexes.values())

    def __getitem__(self, index):
        return self.items[index]
    
    def __setitem__(self, index, value):
        self.items[index] = value

if __name__ == "__main__":
    rand_partition = RandPartition(10, 5)