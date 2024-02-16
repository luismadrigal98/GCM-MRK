import numpy as np
from sklearn.metrics import davies_bouldin_score, pairwise_distances
from consolidate_labels import consolidate_labels

def davies_bouldin_index(data, labels, unassigned_penalty=1.0):
    """ 
    Calculate the Davies-Bouldin Index for a given clustering solution.

    Parameters:
    data (numpy.ndarray): A 2D array where each row is a data point and each column is a feature, shape (n_samples, n_features).
    labels (numpy.ndarray): The predicted labels for each data point, shape (n_samples,). Unassigned elements are labeled as 0.
    unassigned_penalty (float, optional): The penalty factor for unassigned elements. Default is 1.0.

    Returns:
    float: The Davies-Bouldin Index value, lower is better. This is a measure of the average 'similarity' between clusters, where the similarity is a measure that compares the distance between clusters with the size of the clusters themselves.
    """
    
    labels = consolidate_labels(labels)
    labels = np.array(labels)

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