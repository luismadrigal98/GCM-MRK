import sklearn.metrics

def pairwise_d(data, metric='euclidean'):
    """Calculate the pairwise distances between data points using a given metric.

    Parameters:
    data (array-like): The data points to be compared, shape (n_samples, n_features).
    metric (str or callable, optional): The distance metric to use. Default is 'euclidean'.

    Returns:
    array: A 2D array where the entry at [i][j] is the distance between points i and j, shape (n_samples, n_samples).
    """

    distance_matrix = sklearn.metrics.pairwise_distances(data, metric=metric)
    return distance_matrix