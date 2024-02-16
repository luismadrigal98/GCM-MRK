import sklearn.metrics

def pairwise_d(data, metric='euclidean'): ## I can do it using chunks
    """Calculates the pairwise distances between data points using a given metric.

    Args:
        data (array-like): The data points to be compared, shape (n_samples, n_features).
        metric (str or callable, optional): The distance metric to use.
            Default is 'euclidean'. See the documentation of
            `sklearn.metrics.pairwise_distances` for a list of available metrics.

    Returns:
        numpy.ndarray: A distance matrix where the entry at [i, j]
            is the distance between points i and j, shape (n_samples, n_samples).
    """

    distance_matrix = sklearn.metrics.pairwise_distances(data, metric=metric)
    return distance_matrix