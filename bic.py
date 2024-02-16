from sklearn.mixture import GaussianMixture
import numpy as np
from consolidate_labels import consolidate_labels

def bic(Distance_matrix, labels, unassigned_penalty=1.0):
    labels = consolidate_labels(labels)
    labels = np.array(labels)

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
        cluster_filter = labels == label  # Use 'labels' instead of 'assigned_labels'
        gmm = GaussianMixture(n_components=1)
        gmm.fit(Distance_matrix[cluster_filter, :][: , cluster_filter])
        bic = gmm.bic(Distance_matrix[cluster_filter, :][: , cluster_filter])
        total_bic += bic

    if len(unassigned_labels) > 0:
        # Penalty for unassigned elements
        n_unassigned = len(unassigned_labels)
        penalty = unassigned_penalty * (n_unassigned / n_samples)

        # Add penalty to BIC
        total_bic += penalty

    return total_bic