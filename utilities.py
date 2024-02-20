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

    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_filter = labels == label
        if sum(cluster_filter) == 1:
            if all_in_clusters:
                # Find the closest cluster
                distances = Distance_matrix[cluster_filter, :]
                closest_cluster_index = np.argmin(np.min(distances, axis=1))
                indices_to_update = np.where(cluster_filter)[0]  # Get actual indices
                labels[indices_to_update] = labels[closest_cluster_index]  # Assign label
            else:
                labels[cluster_filter] = 0

    return labels