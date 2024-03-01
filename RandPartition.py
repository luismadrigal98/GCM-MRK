import random
from internal_indexes import davies_bouldin_index, bic, aic, silhouette_index
from utilities import handle_singletons

class RandPartition:
    """
    A class to represent a random partition of genes into clusters. An individual is represented by a list of integers, 
    which denotes the assignment of that individual to a given cluster. An individual can be understood, then, as a single partition. 
    Individuals can be initialized randomly, or can be initialized taking as input partitions generated from gene-coexpression analysis software, 
    like Clust or WGCNA (See class InpPartition).
    """

    def __init__(self, element_number, G_max, m, all_in_clusters=True):
        """
        Initialize the instance variables and create a random partition.

        Parameters:
            element_number (int): The number of elements to be partitioned.
            G_max (int): The number of groups for partitioning. Default is 2.
            all_in_clusters (bool): If True, all elements are assigned to a cluster. If False, some elements may not be assigned. Default is True.
        """

        self.element_number = element_number
        self.__init_part(element_number, G_max, m, all_in_clusters)

    def __len__(self):
        """
        Return the length of the chromosome in the individual.

        Returns:
            int: The length of the chromosome.
        """

        return len(self.items)
    
    def __init_part(self, element_number, G_max, m, all_in_clusters):
        """
        Create a random partition of genes into clusters.

        Parameters:
            element_number (int): The number of elements to be partitioned.
            G_max (int): The number of groups for partitioning.
            all_in_clusters (bool): If True, all elements are assigned to a cluster. If False, some elements may not be assigned.
        """

        start = 1 if all_in_clusters else 0
        end = random.randint(start + 1, G_max) if all_in_clusters else random.randint(start + 2, G_max)
        self.items = [random.randint(start, end) for _ in range(element_number)]

    def get_values(self, Data, Distance_matrix, m, unassigned_penalty, DBI, BIC, AIC, SI):
        """
        Calculate different internal measurements of cluster quality.

        Parameters:
            Data (list): The data to be clustered.
            Distance_matrix (list): The distance matrix of the data.
            DBI (bool): If True, calculate the Davies-Bouldin index. Default is True.
            BIC (bool): If True, calculate the Bayesian Information Criterion. Default is True.
            AIC (bool): If True, calculate the Akaike Information Criterion. Default is True.

        Returns:
            tuple: A tuple containing the calculated internal measurements of cluster quality.
        """

        internal_indexes = {}

        if DBI:
            internal_indexes["DBI"] = davies_bouldin_index(Data, self.items, unassigned_penalty)

        if BIC:
            internal_indexes["BIC"] = bic(Distance_matrix, self.items, m, unassigned_penalty)

        if AIC:
            internal_indexes["AIC"] = aic(Distance_matrix, self.items, m, unassigned_penalty)
        
        if SI:
            internal_indexes["SI"] = silhouette_index(Distance_matrix, self.items, m, unassigned_penalty)

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
        self.items[index] = value

if __name__ == "__main__":
    rand_partition = RandPartition(10, 5, 10)
    print(rand_partition.items)