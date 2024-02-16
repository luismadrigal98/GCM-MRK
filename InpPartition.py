"""
This script contains the definition of the InpPartition class. An individual is represented by a list of integers, which denotes the assignment of that individual
to a given cluster. An individual can be understood, then, as a single partition. Individuals can be initialized taking as input partitions those generated from 
other clustering softwares or can be initialized randomly (See class RandPartition).

"""

from davies_bouldin_index import davies_bouldin_index
import pairwise_distances
from bic import bic
from aic import aic
import random
from RandPartition import RandPartition

class InpPartition:
    """
    A class to represent a partition of elements into clusters.

    Attributes:
        instance_count (int): A class variable that counts the number of instances of this class.
    """

    instance_count = 0  # Class variable to keep track of instances

    def __init__(self, element_number, G = 2, input = None, sep = ",", all_in_clusters = True):
        """
        Initializes the instance variables and creates a partition.

        Args:
            element_number (int): The number of elements to be clustered.
            G (int, optional): Defaults to 2. Number of components (clusters).
            input (str, optional): Defaults to None. If provided, should be the path to a file where each line represents a partition.
            sep (str, optional): Defaults to ",". The separator used in the input file.
            all_in_clusters (bool, optional): Defaults to True. Unknown parameter, needs clarification.
        """

        self.__init_part(self, element_number, G, input = None, sep = ",", all_in_clusters = True)
        InpPartition.instance_count += 1  # Increment count when instance is created

    def __len__(self):
        """
        Returns the length of the chromosome in the individual.

        Returns:
            int: The length of the chromosome.
        """

        return len(self.items)
    
    def __init_part(self, element_number, G, input = None, sep = ",", all_in_clusters = True):
        """
        Creates a partition of elements into clusters. The partition can be created from an input file or randomly.

        Args:
            element_number (int): The number of elements to be partitioned.
            G (int): Number of components (clusters).
            input (str, optional): Defaults to None. If provided, should be the path to a file where each line represents a partition.
            sep (str, optional): Defaults to ",". The separator used in the input file.
            all_in_clusters (bool, optional): Defaults to True. Unknown parameter, needs clarification.

        Returns:
            list: A partition of elements into clusters.
        """

        for entry in range(InpPartition.number_of_partitions):

            if input == None:
                return RandPartition(element_number, G, all_in_clusters = all_in_clusters)

            with open(input, "r") as file:
                try:
                    for _ in range(InpPartition.instance_count):  # Skip lines already read by previous instances
                        next(file)
                except StopIteration:
                    print("You are creating more instances than the number of lines in the input file. The rest of the partitions will be generated at random")
                line = file.readline()
                if not line:  # We've reached the end of the file
                    break
                part = line.replace("\n", "").split(sep)
                partition = [elem for elem in part]

            if not partition:  # If no partition was read from the file, generate a random

                rand_partition = RandPartition(len(InpPartition.all_partitions[0]), G = InpPartition.max_number_of_elements, 
                                            all_in_clusters = all_in_clusters)
                partition = rand_partition

            if all_in_clusters:
                
                original_list = partition
                
                for index, value in enumerate(original_list):
                    minimum = min(original_list) + 1
                    maximum = max(original_list)
                    
                    if value == 0:
                        original_list[index] = random.randint(minimum, maximum)
                
                self.items = original_list
            
            else:
                self.items = partition

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
    inp_partition = InpPartition(10, 5)