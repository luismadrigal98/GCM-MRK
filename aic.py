from sklearn.mixture import GaussianMixture
import numpy as np

def aic(Distance_matrix, labels, unassigned_penalty=1.0):
    """Calculate the Akaike Information Criterion (AIC) for a given clustering solution.

    The AIC is a metric that evaluates the quality of a clustering solution based on the likelihood of the data given the model and the number of parameters in the model. Lower AIC values indicate better clustering because they represent models that fit the data well and are not too complex. This function also adds a penalty term for unassigned elements, which are data points that are not assigned to any cluster.

    Parameters:
    Distance_matrix (array-like): A 2D array where the entry at [i][j] is the distance between points i and j, shape (n_samples, n_samples).
    labels (array-like): The predicted labels for each data point, shape (n_samples,). Unassigned elements are labeled as 0.
    unassigned_penalty (float, optional): The penalty factor for unassigned elements. Default is 1.0.

    Returns:
    float: The AIC value, lower is better.
    """
    
    labels = np.array(labels)

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
        # Penalty for unassigned elements
        n_unassigned = len(unassigned_labels)
        penalty = unassigned_penalty * (n_unassigned / n_samples)

        # Add penalty to AIC
        total_aic += penalty

    return total_aic