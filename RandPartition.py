import random
import numpy as np
from internal_indexes import likelihood_calculator, calculate_aic, calculate_bic

class RandPartition:
    """
    A class to represent a random partition of genes into clusters. An individual is represented by a list of integers, 
    which denotes the assignment of that individual to a given cluster. An individual can be understood, then, as a single partition. 
    Individuals can be initialized randomly, or can be initialized taking as input partitions generated from gene-coexpression analysis software, 
    like Clust or WGCNA (See class InpPartition).
    """

    def __init__(self, element_number, G_max, all_in_clusters):
        """
        Initialize the instance variables and create a random partition.

        Parameters:
            element_number (int): The number of elements to be partitioned.
            G_max (int): The number of groups for partitioning. Default is 2.
            all_in_clusters (bool): If True, all elements are assigned to a cluster. If False, some elements may not be assigned. Default is True.
        """

        if not isinstance(element_number, int) or element_number <= 0:
            raise ValueError("Element_number must be a positive integer")
        if not isinstance(G_max, int) or G_max <= 0:
            raise ValueError("G_max must be a positive integer")
        if not isinstance(all_in_clusters, bool):
            raise ValueError("All_in_clusters must be a boolean")

        self.element_number = element_number
        self.__init_part(element_number, G_max, all_in_clusters)

    def __len__(self):
        """
        Return the length of the chromosome in the individual.

        Returns:
            int: The length of the chromosome.
        """

        return len(self.items)
    
    def __init_part(self, element_number, G_max, all_in_clusters):
        """
        Create a random partition of genes into clusters.

        Parameters:
            element_number (int): The number of elements to be partitioned.
            G_max (int): The number of groups for partitioning.
            all_in_clusters (bool): If True, all elements are assigned to a cluster. If False, some elements may not be assigned.
        """

        start = 1 if all_in_clusters else 0
        end = random.randint(start + 1, G_max) if all_in_clusters else random.randint(start + 2, G_max)
        if start != end:
            self.items = [random.randint(start, end) for _ in range(element_number)]
        else:
            self.items = [start for _ in range(element_number)]

    def get_values(self, cor_matrix, Log_Likelihood, BIC, AIC):
        """
        Calculate different internal measurements of cluster quality.

        Parameters:
            cor_matrix (np.ndarray): The correlation matrix of the data.
            Log_Likelihood (bool): If True, calculate the log-likelihood of the clustering solution.
            BIC (bool): If True, calculate the Bayesian Information Criterion (BIC) of the clustering solution.
            AIC (bool): If True, calculate the Akaike Information Criterion (AIC) of the clustering solution.

        Returns:
            tuple: A tuple containing the calculated internal measurements of cluster quality.
        """
        
        internal_indexes = {}
        labels = self.items[:]

        if Log_Likelihood:
            internal_indexes["Log_Likelihood"] = likelihood_calculator(cor_matrix, labels)

        if BIC and Log_Likelihood:
            internal_indexes["BIC"] = calculate_bic(internal_indexes["Log_Likelihood"], len(np.unique(np.array(labels))), len(labels))
        elif not BIC:
            pass
        else:
            raise ValueError("BIC not calculated. It requires the calculation of the log-likelihood. Please set Log_Likelihood to True.")

        if AIC and Log_Likelihood:
            internal_indexes["AIC"] = calculate_aic(internal_indexes["Log_Likelihood"], len(np.unique(np.array(labels))))
        elif not AIC:
            pass
        else:
            raise ValueError("AIC not calculated. It requires the calculation of the log-likelihood. Please set Log_Likelihood to True.")

        return tuple(internal_indexes.values())

    def __getitem__(self, index):
        """
            Get the item at the specified index.

        Parameters:
            index (int): The index of the item.

        Returns:
            int: The item at the specified index.
        """
        return self.items[index]
    
    def __setitem__(self, index, value):
        """
        Set the item at the specified index to the specified value.

        Parameters:
            index (int): The index of the item.
            value (int): The value to set the item to.
        """

        self.items[index] = value

if __name__ == "__main__":
    rand_partition = RandPartition(10, 5, True)
    print(rand_partition.items)