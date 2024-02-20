from sklearn.mixture import GaussianMixture
import numpy as np
from utilities import consolidate_labels, handle_singletons
from sklearn.metrics import davies_bouldin_score, pairwise_distances

def davies_bouldin_index(data, labels, unassigned_penalty):
    """
    Calculate the Davies-Bouldin Index for a given clustering solution.

    Parameters:
        data (numpy.ndarray): A 2D array where each row is a data point and each column is a feature, shape (n_samples, n_features).
        labels (numpy.ndarray): The predicted labels for each data point, shape (n_samples,). Unassigned elements are labeled as 0.
        unassigned_penalty (float, optional): The penalty factor for unassigned elements. Default is 1.0.

    Returns:
        float: The Davies-Bouldin Index value, lower is better. This is a measure of the average 'similarity' between clusters, where the similarity is a measure that compares the distance between clusters with the size of the clusters themselves.
    """
    try:
        # Input validation
        assert isinstance(data, np.ndarray), "Data must be a NumPy array"

        labels = consolidate_labels(labels)
        labels = np.array(labels)

        assert np.issubdtype(labels.dtype, np.integer), "Labels must be integers"

        n_samples = len(labels)

        if 0 in labels:
            n_unassigned = len(labels[labels == 0])  # count unassigned elements
        else:
            n_unassigned = 0
        
        filter = np.array([i != 0 for i in labels])
        assigned_labels = labels[filter]  # filter out unassigned elements

        db_index = davies_bouldin_score(data[filter, :], labels=assigned_labels)

        # Penalty for unassigned elements
        if n_unassigned > 1:
            unassigned_distances = np.mean(pairwise_distances(data[labels[labels == 0], :]))
            db_index += unassigned_penalty * (n_unassigned / n_samples) * unassigned_distances
        elif n_unassigned ==1:
            db_index += unassigned_penalty * (n_unassigned / n_samples)

        return db_index
    except AssertionError as e:
        raise ValueError(f"Invalid input: {e}") from e

def aic(Distance_matrix, labels, m, unassigned_penalty):
    """Calculate the Akaike Information Criterion (AIC) for a given clustering solution.

    The AIC is a metric that evaluates the quality of a clustering solution based on the likelihood of the data given the model and the number of parameters in the model. Lower AIC values indicate better clustering because they represent models that fit the data well and are not too complex. This function also adds a penalty term for unassigned elements, which are data points that are not assigned to any cluster.

    Parameters:
        Distance_matrix (array-like): A 2D array where the entry at [i][j] is the distance between points i and j, shape (n_samples, n_samples).
        labels (array-like): The predicted labels for each data point, shape (n_samples,). Unassigned elements are labeled as 0.
        m (int): An integer that represent the number of feature that were used for assessing the distance matrix.
        unassigned_penalty (float, optional): The penalty factor for unassigned elements. Default is 1.0.

    Returns:
        float: The AIC value, lower is better.

    Raises:
        ValueError: If Distance_matrix is not a square matrix or contains non-numeric values.
        TypeError: If labels are not an array-like or contain non-integer values.
        ValueError: If all labels are unassigned (0).
    """
    try:
        # Input validation
        assert isinstance(Distance_matrix, np.ndarray), "Distance_matrix must be a NumPy array"
        assert Distance_matrix.ndim == 2 and Distance_matrix.shape[0] == Distance_matrix.shape[1], "Distance_matrix must be a square matrix"
        assert np.issubdtype(Distance_matrix.dtype, np.number), "Distance_matrix must contain numeric values"

        labels = consolidate_labels(labels)
        labels = np.array(labels)

        assert np.issubdtype(labels.dtype, np.integer), "Labels must be integers"

        n_samples = len(labels)

        # Filter out unassigned elements
        filter = labels != 0
        assigned_labels = labels[filter]
        unassigned_labels = labels[labels == 0]

        # Number of clusters
        unique_labels = np.unique(assigned_labels)

        total_aic = 0

        # Fit a separate GMM and calculate AIC for each cluster
        for label in unique_labels:
            cluster_filter = labels == label  # Use 'labels' instead of 'assigned_labels'
            gmm = GaussianMixture(n_components=1)
            gmm.fit(Distance_matrix[cluster_filter, :][: , cluster_filter])
            aic = gmm.aic(Distance_matrix[cluster_filter, :][: , cluster_filter])
            total_aic += aic

        if len(unassigned_labels) > 0:
            # Penalty based on average distance to nearest cluster
            average_distance = np.mean(np.min(Distance_matrix[unassigned_labels], axis=1))
            max_distance = np.max(Distance_matrix)  # Normalize by maximum distance
            distance_penalty = average_distance / max_distance * unassigned_penalty

            # Data-driven penalty (adjust constants as needed)
            data_driven_penalty = len(unassigned_labels) * np.log(n_samples) / (n_samples * m)

            total_penalty = distance_penalty + data_driven_penalty
            total_aic += unassigned_penalty * total_penalty

        return total_aic
    
    except AssertionError as e:
        raise ValueError(f"Invalid input: {e}") from e
    
from sklearn.mixture import GaussianMixture
import numpy as np
from utilities import consolidate_labels

def bic(Distance_matrix, labels, m, unassigned_penalty):
    """
    This function calculates the Bayesian Information Criterion (BIC) for a given distance matrix and labels.
    
    Parameters:
        Distance_matrix (numpy.ndarray): A square matrix representing the distances between data points.
        labels (list or numpy.ndarray): A list or array of labels corresponding to the data points. Unassigned data points should be labeled as 0.
        m (int): An integer that represent the number of feature that were used for assessing the distance matrix.
        unassigned_penalty (float, optional): The penalty factor for unassigned elements. Default is 1.0.
    
    Returns:
        total_bic (float): The total BIC score for the given data and labels, including penalties for unassigned elements.
    
    Raises:
        ValueError: If Distance_matrix is not a square matrix or contains non-numeric values.
        ValueError: If labels are not an array-like or contain non-integer values.
        ValueError: If all labels are unassigned (0).

    Note:
        The function fits a separate Gaussian Mixture Model (GMM) for each cluster (unique label) and calculates the BIC for each. 
        If there are unassigned elements (label 0), a penalty is added to the total BIC. The penalty is calculated as the product of the unassigned_penalty parameter and the proportion of unassigned elements.
    """
    try:
        # Input validation
        assert isinstance(Distance_matrix, np.ndarray), "Distance_matrix must be a NumPy array"
        assert Distance_matrix.ndim == 2 and Distance_matrix.shape[0] == Distance_matrix.shape[1], "Distance_matrix must be a square matrix"
        assert np.issubdtype(Distance_matrix.dtype, np.number), "Distance_matrix must contain numeric values"

        labels = consolidate_labels(labels)  # Ensure consolidated labels
        labels = np.array(labels)

        assert np.issubdtype(labels.dtype, np.integer), "Labels must be integers"

        n_samples = len(labels)

        # Filter out unassigned elements
        filter = labels != 0
        assigned_labels = labels[filter]
        unassigned_labels = labels[labels == 0]

        # Number of clusters
        unique_labels = np.unique(assigned_labels)

        total_bic = 0

        # Fit a separate GMM and calculate BIC for each cluster
        for label in unique_labels:
            cluster_filter = labels == label
            gmm = GaussianMixture(n_components=1)
            gmm.fit(Distance_matrix[cluster_filter, :][: , cluster_filter])
            bic = gmm.bic(Distance_matrix[cluster_filter, :][: , cluster_filter])
            total_bic += bic

        if len(unassigned_labels) > 0:
            # Penalty based on average distance to nearest cluster
            average_distance = np.mean(np.min(Distance_matrix[unassigned_labels], axis=1))
            max_distance = np.max(Distance_matrix)  # Normalize by maximum distance
            distance_penalty = average_distance / max_distance * unassigned_penalty

            # Data-driven penalty (adjust constants as needed)
            data_driven_penalty = len(unassigned_labels) * np.log(n_samples) / (n_samples * m)

            total_penalty = distance_penalty + data_driven_penalty
            total_bic += unassigned_penalty * total_penalty

        return total_bic
    
    except AssertionError as e:
        raise ValueError(f"Invalid input: {e}") from e