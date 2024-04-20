"""
This script contains the definition of the InpPartition class. An individual is represented by a list of integers, which denotes the assignment of that individual
to a given cluster. An individual can be understood, then, as a single partition. Individuals can be initialized taking as input partitions those generated from 
other clustering softwares or can be initialized randomly (See class RandPartition).

"""

from internal_indexes import likelihood_calculator, calculate_aic, calculate_bic
import random

class InpPartition:
    """
    A class to represent a partition of elements into clusters.

    Attributes:
        instance_count (int): A class variable that counts the number of instances of this class.
    """

    instance_count = 0  # Class variable to keep track of instances

    def __init__(self, element_number, G_max, input, sep, all_in_clusters):
        """
        Initializes the instance variables and creates a partition.

        Args:
            element_number (int): The number of elements to be clustered.
            G_max (int, optional): Number of components (clusters).
            input (str, optional): Defaults to None. If provided, should be the path to a file where each line represents a partition.
            sep (str, optional): Defaults to ",". The separator used in the input file.
            all_in_clusters (bool, optional): Defaults to True. Unknown parameter, needs clarification.
        """

        if not isinstance(element_number, int) or element_number <= 0:
            raise ValueError("Element_number must be a positive integer")
        if not isinstance(G_max, int) or G_max <= 0:
            raise ValueError("G_max must be a positive integer")
        if input is not None and not isinstance(input, str):
            raise ValueError("input_file must be a string")
        if not isinstance(sep, str):
            raise ValueError("sep must be a string")
        if not isinstance(all_in_clusters, bool):
            raise ValueError("All_in_clusters must be a boolean")

        self.__init_part(element_number, G_max, input, sep, all_in_clusters)
        InpPartition.instance_count += 1  # Increment count when instance is created

    def __len__(self):
        """
        Returns the length of the chromosome in the individual.

        Returns:
            int: The length of the chromosome.
        """

        return len(self.items)
    
    def __init_part(self, element_number, G_max, input, sep, all_in_clusters):
        """
        Creates a partition of elements into clusters. The partition can be created from an input file or randomly.

        Args:
            element_number (int): The number of elements to be partitioned.
            G_max (int): Number of components (clusters).
            input (str, optional): Defaults to None. If provided, should be the path to a file where each line represents a partition.
            sep (str, optional): Defaults to ",". The separator used in the input file.
            all_in_clusters (bool, optional): Defaults to True. Unknown parameter, needs clarification.

        Returns:
            list: A partition of elements into clusters.
        """
        
        partition_from_input = False  # Initialize partition_from_input
        
        if input is not None:
            with open(input, "r") as file:
                try:
                    for _ in range(InpPartition.instance_count):  # Skip lines already read by previous instances
                        next(file)
                except StopIteration:
                    print("You are creating more instances than the number of lines in the input file. The rest of the partitions will be generated at random")
                line = file.readline()
                if not line:  # We've reached the end of the file
                    partition_from_input = False
                else:
                    part = line.replace("\n", "").split(sep)
                    partition = [elem for elem in part]

        if not partition_from_input:  # If no partition was read from the file, generate a random  <<<<<<< This is erasing 

            start = 1 if all_in_clusters else 0
            end = random.randint(start + 1, G_max) if all_in_clusters else random.randint(start + 2, G_max)
            rand_partition = [random.randint(start, end) for _ in range(element_number)]
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

    def get_values(self, Data, cor_matrix, Distance_matrix, m, unassigned_penalty, Log_Likelihood, BIC, 
                   AIC, DBI, SI, CHI):
        """
        Calculates different internal measurements of cluster quality.

        Args:
            Data (array-like): The dataset for which to calculate the measurements.
            DBI (bool, optional): Whether to calculate the Davies-Bouldin Index. Defaults to True.
            BIC (bool, optional): Whether to calculate the Bayesian Information Criterion. Defaults to True.
            AIC (bool, optional): Whether to calculate the Akaike Information Criterion. Defaults to True.

        Returns:
            tuple: A tuple of the calculated measurements.
        """

        internal_indexes = {}
        labels = self.items[:]

        if Log_Likelihood:
            internal_indexes["Log_Likelihood"] = likelihood_calculator(cor_matrix, labels)

        if BIC:
            internal_indexes["BIC"] = calculate_bic(Distance_matrix, labels, m, unassigned_penalty)

        if AIC:
            internal_indexes["AIC"] = calculate_aic(Distance_matrix, labels, m, unassigned_penalty)

        if DBI:
            internal_indexes["DBI"] = davies_bouldin_index(Data, labels, unassigned_penalty)
        
        if SI:
            internal_indexes["SI"] = silhouette_index(Distance_matrix, labels, m, unassigned_penalty)

        if CHI:
            internal_indexes["CHI"] = calinski_harabasz_index(Data, labels, m, unassigned_penalty)

        return tuple(internal_indexes.values())

    def __getitem__(self, index):
        """
        Gets the item at the specified index.

        Args:
            index (int): The index of the item to get.

        Returns:
            The item at the specified index.
        """

        return self.items[index]
    
    def __setitem__(self, index, value):
        """
        Sets the item at the specified index to the specified value.

        Args:
            index (int): The index of the item to set.
            value: The value to set the item to.
        """
        
        self.items[index] = value

if __name__ == "__main__":
    inp_partition = InpPartition(10, 5)
    print(inp_partition.items)