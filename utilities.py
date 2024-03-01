import sklearn.metrics
import numpy as np
import math as m

def pairwise_d(data, metric='euclidean', dis_type = None): ## I can do it using chunks
    """Calculates the pairwise distances between data points using a given metric.

    Args:
        data (array-like): The data points to be compared, shape (n_samples, n_features). Data msut be standardized to implement
            the pearson-based metric, because the distance matrix will be transformed to an eculidean space for using the validation
            indexes. This is possible only if the data is centered and scaled (normalized).
        metric (str or callable, optional): The distance metric to use.
            Default is 'euclidean'. See the documentation of
            `sklearn.metrics.pairwise_distances` for a list of available metrics.
        dis_type (str, only needed if metric is 'pearson'): if Pearson-based distance is selected as metric, the user must specify the kind of distance he wants to determine,
            whether a signed (considering the sign of the correlation) or an unsigend one. Expected values:
            dis_type = 'signed' or dis_type = 'unsigned'

    Returns:
        numpy.ndarray: A distance matrix where the entry at [i, j]
            is the distance between points i and j, shape (n_samples, n_samples).
    """
    
    assert isinstance(data, np.ndarray), "Data must be a numpy array"
    assert data.ndim == 2, "Data must be a 2D array (n_samples, n_features)"
    assert np.all(data.shape[1] == data[0].shape[0]), "All data points must have the same number of features"

    if isinstance(metric, str):
        supported_metrics = ['euclidean', 'pearson']
        assert metric in supported_metrics, f"metric must be one of {supported_metrics}"
    else:
        assert callable(metric), "metric must be a string or a callable function"

    if metric == "pearson":
        assert dis_type in ['signed', 'unsigned'], "dis_type must be 'signed' or 'unsigned' for Pearson metric"

    try:
        if metric == "euclidean":
            distance_matrix = sklearn.metrics.pairwise_distances(data, metric=metric)
            return distance_matrix
        
        elif metric == "pearson":

            correlation_matrix = np.corrcoef(data)

            m = correlation_matrix.shape[0]

            if dis_type == 'signed':
                euclidean_distance = m.sqrt(2 * m * (1 - correlation_matrix))
            if dis_type == 'unsigned':
                euclidean_distance = m.sqrt(2 * m * (1 - abs(correlation_matrix)))

            return euclidean_distance
    
    except: 
        raise ValueError("Metric should be one of the followings: 'euclidean' or 'pearson'.")

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
    
def handle_singletons(Distance_matrix, labels, all_in_clusters):
    """Assigns singletons to the closest cluster if all_in_clusters is True."""

    labels = np.array(labels)

    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_filter = labels == label
        opossite_filter = ~ cluster_filter
        if sum(cluster_filter) == 1:
            if all_in_clusters:
                # Find the closest cluster
                distances = Distance_matrix[cluster_filter, :]
                distances = distances[distances > 0]
                labels_singleton_out = labels[opossite_filter]
                closest_cluster_index = np.argmin(distances)
                indices_to_update = np.where(cluster_filter)  # Get actual indices
                labels[indices_to_update[0]] = labels_singleton_out[closest_cluster_index]  # Assign label
            else:
                labels[cluster_filter] = 0

    return labels.tolist()

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

def normalize_data(data):
    """Normalizes the data to have zero mean and unit variance.

    Args:
        data (numpy.ndarray): The data to be normalized, shape (n_samples, n_features).

    Returns:
        numpy.ndarray: The normalized data, shape (n_samples, n_features).
    """
    assert isinstance(data, np.ndarray), "Data must be a numpy array"
    assert data.ndim == 2, "Data must be a 2D array (n_samples, n_features)"

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