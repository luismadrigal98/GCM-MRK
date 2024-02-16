import numpy as np

def davies_bouldin_index(Distance_matrix, labels, unassigned_penalty=1.0):
    """Calculate the Davies-Bouldin Index for a given clustering solution.

    The Davies-Bouldin Index (DBI) is a metric that evaluates the quality of a clustering solution based on the ratio of within-cluster distances to between-cluster distances. Lower DBI values indicate better clustering because they represent clusters that are well-separated and compact. This function also adds a penalty term for unassigned elements, which are data points that are not assigned to any cluster.

    Parameters:
    Distance_matrix (array-like): A 2D array where the entry at [i][j] is the distance between points i and j, shape (n_samples, n_samples).
    labels (array-like): The predicted labels for each data point, shape (n_samples,). Unassigned elements are labeled as 0.
    unassigned_penalty (float, optional): The penalty factor for unassigned elements. Default is 1.0.

    Returns:
    float: The Davies-Bouldin Index value, lower is better.
    """
        
    n_samples = len(labels)
    n_unassigned = len(labels[labels == 0])  # count unassigned elements
    assigned_labels = labels[labels != 0]  # filter out unassigned elements

    n_cluster = len(np.unique(assigned_labels))
    cluster_distance = [np.mean(Distance_matrix[assigned_labels==i]) for i in range(1, n_cluster + 1)]
    centroid_distance = [np.mean(Distance_matrix[assigned_labels==i][:, assigned_labels==j]) for i in range(1, n_cluster+1) for j in range(1, n_cluster + 1) if i != j]
    db_index = np.mean([max((cluster_distance[i] + cluster_distance[j]) / centroid_distance[i*n_cluster+j] for j in range(n_cluster) if i != j) for i in range(n_cluster)])

    # Penalty for unassigned elements
    if n_unassigned > 0:
        unassigned_distances = np.mean(Distance_matrix[labels == 0])
        db_index += unassigned_penalty * (n_unassigned / n_samples) * unassigned_distances

    return db_index