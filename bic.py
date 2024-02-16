from sklearn.mixture import GaussianMixture
import numpy as np

def bic(Distance_matrix, labels, unassigned_penalty=1.0):
    """Calculate the Bayesian Information Criterion (BIC) for a given clustering solution.

    The BIC is a metric that evaluates the quality of a clustering solution based on the likelihood of the data given the model and the complexity of the model. Lower BIC values indicate better clustering because they represent models that fit the data well and are not too complex. This function also adds a penalty term for unassigned elements, which are data points that are not assigned to any cluster.

    Parameters:
    Distance_matrix (array-like): A 2D array where the entry at [i][j] is the distance between points i and j, shape (n_samples, n_samples).
    labels (array-like): The predicted labels for each data point, shape (n_samples,). Unassigned elements are labeled as 0.
    unassigned_penalty (float, optional): The penalty factor for unassigned elements. Default is 1.0.

    Returns:
    float: The BIC value, lower is better.
    """

    # Filter out unassigned elements
    assigned_labels = labels[labels != 0]
    unassigned_labels = labels[labels == 0]

    # Number of clusters
    n_clusters = len(np.unique(assigned_labels))

    # Fit a Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_clusters)
    gmm.fit(Distance_matrix)

    # Calculate BIC
    bic = gmm.bic(Distance_matrix)

    # Penalty for unassigned elements
    n_samples = len(labels)
    n_unassigned = len(unassigned_labels)
    penalty = unassigned_penalty * (n_unassigned / n_samples)

    # Add penalty to BIC
    bic += penalty

    return bic