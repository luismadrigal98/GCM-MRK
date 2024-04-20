import numpy as np
import math as m
from scipy.stats import entropy

def consolidate_labels(labels):

    """Consolidates labels by assigning consecutive integers starting from 0,
    handling the presence of the label 0.

    Args:
        labels (array-like or list): An array or list of labels.

    Returns:
        list: A list of consolidated labels with the same shape as the input.

    Raises:
        ValueError: If any label is not an integer.

    Examples:
        >>> consolidate_labels([6, 2, 0, 2, 1])
        [3, 2, 0, 2, 1]

        >>> consolidate_labels([3, 2, 1, 0])
        [3, 2, 1, 0]

        >>> consolidate_labels([9, 9, 2, 2, 4])
        [3, 3, 1, 1, 2]
    """

    try:
        if 0 in labels:
            dictionary_labels = {old: new for new, old in enumerate(sorted(set(labels)), start=0)}

        else:
            dictionary_labels = {old: new for new, old in enumerate(sorted(set(labels)), start=1)}

        new_labels = [dictionary_labels[label] for label in labels]

        return new_labels
    except:
        raise TypeError("Labels must be integers.")

def write_results_to_file(out, log, file_path):
    """
    Writes the results of the GCM_MRK function to a text file.

    Args:
        out (dict): The output of the GCM_MRK function. It's a dictionary containing 
                    'Partitions' and 'Fitness' as keys.
        log (deap.tools.Logbook): The logbook that contains the statistics of the 
                                  generations.
        file_path (str): The path to the file where the results should be written.

    Returns:
        None
    """    
    with open(file_path, "w") as file:
        file.write("****************************************************\n")
        file.write("Final partitions\n")
        file.write("____________________________________________________\n")
        file.write('\n')
        for lst in out["Partitions"]:
            file.write(f'{lst}\n')
        file.write('\n')
        file.write("****************************************************\n")
        file.write("Fitness values\n")
        file.write("____________________________________________________\n")
        file.write('\n')
        for lst in out["Fitness"]:
            file.write(f'{lst}\n')
        file.write('\n')
        file.write("****************************************************\n")
        file.write("Log\n")
        file.write("____________________________________________________\n")
        file.write('\n')
        for lst in log:
            file.write(f'{lst}\n')

def normalize_data(data, by_sample = False):
    """Normalizes the data to have zero mean and unit variance.

    Args:
        data (numpy.ndarray): The data to be normalized, shape (n_samples, n_features).

    Returns:
        numpy.ndarray: The normalized data, shape (n_samples, n_features).
    """
    assert isinstance(data, np.ndarray), "Data must be a numpy array"
    assert data.ndim == 2, "Data must be a 2D array (n_samples, n_features)"

    if by_sample:
        mean = np.mean(data, axis=1)
        std = np.std(data, axis=1)
        
        return (data - mean[:, np.newaxis]) / std[:, np.newaxis]

    else:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        return (data - mean) / std

def range_per_index(individuals):
    """
    Calculate the range (max - min) for each index across a list of individual fitness values.

    Args:
        individuals (list): A list of individuals, where each individual is a tuple of values.

    Returns:
        list: A list of ranges, one for each index in the individuals' tuples.

    Example:
        >>> range_per_index([(1, 2, 3), (4, 5, 6), (7, 8, 9)])
        [6, 6, 6]
    """
    # Transpose the list of individuals to group values by index
    values_by_index = list(zip(*individuals))
    # Calculate the range for each index
    return [np.max(values) - np.min(values) for values in values_by_index]

def population_entropy(individuals):
    """
    Calculate the entropy of a population in a genetic algorithm.

    This function calculates the entropy based on the frequency of each label (cluster assignment) in the population.
    A higher entropy indicates a more diverse population.

    Parameters:
    population (list): A list of individuals in the population. Each individual is an object with an 'items' attribute 
                       that is a list of integers representing cluster assignments.

    Returns:
    float: The entropy of the population.
    """
    # Flatten the population to get a list of all labels
    all_labels = [label for individual in individuals for label in individual.items]
    # Calculate the frequency of each label
    label_freqs = np.bincount(all_labels)
    # Calculate and return the entropy
    return entropy(label_freqs)

def Pearson_correlation(data, rowvar=True):
    """
    Calculate the Pearson correlation coefficient between all pairs of sample vectors in a dataset.

    Args:
        data (numpy.ndarray): The input data. If rowvar is True, shape should be (n_samples, n_features). 
                              If rowvar is False, shape should be (n_features, n_samples).
        rowvar (bool): If True (default), the correlation is assesed between observations (rows).

    Returns:
        numpy.ndarray: The Pearson correlation coefficient matrix. If rowvar is True, shape is (n_samples, n_samples). 
                       If rowvar is False, shape is (n_features, n_features).
    """
    # Calculate the correlation matrix
    correlation_matrix = np.corrcoef(data, rowvar=rowvar)
    return correlation_matrix