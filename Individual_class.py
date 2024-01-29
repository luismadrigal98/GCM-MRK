"""
This script have the definition of the individual class. An individual is represented by a list of integers, which denotates the assignation of that individual
to a given cluster. An individual can be understood, then, as a single partition. Individuals can be initialized randomly, or can be initialized taking as
input partitions generated from gene-coexpression analysis softwares, like Clust or WGCNA

"""

import random

class RandPartition:

    def __init__(self, geneNumber=0, G=0, all_in_clusters=True):

        ## Initializing the instance variables
        self.items = []
        self.geneNumber = geneNumber

        ## Initializing one instance of RandPartition
        self.__initPart(geneNumber, G, all_in_clusters)

    def __len__(self):

        ## Method for seeing the length of the chromosome in the individual
        return len(self.items)
    
    def __initPart(self, geneNumber, G, all_in_clusters=True):

        ## Method for creating a random partition of assignation to clusters, given the length of the chromosome (geneNumber) and the number of clusters (G).
        if all_in_clusters:
            start = 1
        else:
            start = 0

        self.items = [random.randint(start, G) for i in range(geneNumber)]

    def __getValuesInt(self):
        
        # This method will contain the chunk of code for calculating different internal measurements of cluster quality. The parameters of the method will be
        # booleans for enabling or disabling the calculations of those indexes.
        pass  # To be implemented


def main():
    ## Initializing an instance of the class:
    RandIndividual = RandPartition(10, 5)

if __name__ == "__main__":
    main()